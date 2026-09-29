"""Local dopamine-gated KC->MBON plasticity for the frozen FlyBrain.

The simulator's connectome remains the scaffold.  This module keeps a dynamic
modulation value only for existing KC->MBON synapses and writes those values to
the CUDA CSR matrix before the next network step.  All other synapses, neuron
voltages, and readout weights are untouched.

Two gates are intentionally explicit:

``biological``
    DAN activity is measured from the simulated PAM/PPL/PPM neurons and mixed
    into each MBON using its actual DAN->MBON connectome edges.  PAM is treated
    as positive valence, PPL as negative, and PPM as a half-strength positive
    component because the available annotation does not resolve its valence.

``engineering``
    A caller supplies a scalar teaching signal (for example a next-character
    label or prediction error).  It is projected through the same local MBON
    axis but is never reported as biological evidence.
"""

from __future__ import annotations

import numpy as np


class KCMBONPlasticity:
    """Dynamic KC->MBON synapses with an eligibility trace.

    Parameters are deliberately small and interpretable.  ``eta`` is the
    per-token modulation step, ``eligibility_tau`` controls short-term credit
    assignment, and ``weight_tau`` provides slow passive forgetting.  Modulation
    is clipped so an existing edge cannot flip sign or become unbounded.
    """

    def __init__(self, brain, mode: str = "biological", eta: float = 0.02,
                 eligibility_tau: float = 0.5, weight_tau: float = 8.0,
                 max_modulation: float = 0.9,
                 eligibility_mix: float = 0.0):
        if brain.device != "cuda":
            raise ValueError("KCMBONPlasticity currently requires a CUDA FlyBrain")
        if mode not in ("biological", "engineering"):
            raise ValueError("mode must be biological or engineering")
        if eta < 0 or eligibility_tau <= 0 or weight_tau <= 0:
            raise ValueError("eta must be non-negative and time constants must be positive")
        if not 0.0 <= eligibility_mix <= 1.0:
            raise ValueError("eligibility_mix must lie in [0, 1]")
        self.brain = brain
        self.mode = mode
        self.eta = float(eta)
        self.eligibility_tau = float(eligibility_tau)
        self.weight_tau = float(weight_tau)
        self.max_modulation = float(max_modulation)
        self.eligibility_mix = float(eligibility_mix)
        self.xp = brain.xp

        ct = np.asarray(brain.cell_type).astype(str)
        self.kc_ids = np.flatnonzero(np.char.find(ct, "KC") >= 0).astype(np.int64)
        self.mbon_ids = np.flatnonzero(np.char.find(ct, "MBON") >= 0).astype(np.int64)
        self.dan_ids = np.flatnonzero(
            (np.char.find(ct, "PAM") >= 0) |
            (np.char.find(ct, "PPL") >= 0) |
            (np.char.find(ct, "PPM") >= 0)
        ).astype(np.int64)
        if not len(self.kc_ids) or not len(self.mbon_ids):
            raise ValueError("connectome has no KC or MBON population")

        # The live CUDA matrix is CSR (rows = postsynaptic neurons).  Locate
        # only existing KC->MBON entries in that storage order.
        csr = brain._W
        indptr = self.xp.asnumpy(csr.indptr)
        indices = self.xp.asnumpy(csr.indices)
        edge_pos, edge_kc, edge_mbon = [], [], []
        mbon_slot = {int(n): i for i, n in enumerate(self.mbon_ids)}
        kc_set = set(int(n) for n in self.kc_ids)
        for post in self.mbon_ids:
            lo, hi = int(indptr[post]), int(indptr[post + 1])
            for pos in range(lo, hi):
                pre = int(indices[pos])
                if pre in kc_set:
                    edge_pos.append(pos)
                    edge_kc.append(pre)
                    edge_mbon.append(mbon_slot[int(post)])
        if not edge_pos:
            raise ValueError("connectome has no KC->MBON edges")
        self.edge_pos = np.asarray(edge_pos, np.int64)
        edge_kc = np.asarray(edge_kc, np.int64)
        # Keep the presynaptic identity beside the CSR offsets.  A sparse
        # backend must never be allowed to reorder rows after this point: an
        # offset-only cache can otherwise write a modulation value into a
        # different biological edge while all aggregate norms still look
        # plausible.
        self.edge_pre_ids = edge_kc.copy()
        self.edge_kc_slot = np.searchsorted(self.kc_ids, edge_kc).astype(np.int32)
        self.edge_mbon_slot = np.asarray(edge_mbon, np.int32)
        self.edge_pos_gpu = self.xp.asarray(self.edge_pos)
        self.edge_kc_slot_gpu = self.xp.asarray(self.edge_kc_slot)
        self.edge_mbon_slot_gpu = self.xp.asarray(self.edge_mbon_slot)
        self.w0_gpu = csr.data[self.edge_pos_gpu].copy()
        self.modulation = self.xp.zeros(len(self.edge_pos), dtype=self.xp.float32)
        self.kc_eligibility = self.xp.zeros(len(self.kc_ids), dtype=self.xp.float32)
        # Feature sketches are sampled once per token.  Keep their fixed hash
        # assignments and GPU work buffers alive instead of rebuilding them
        # (and their CUDA kernels' temporary arrays) on every sample.
        self._edge_hash_cache = {}
        self._edge_feature_cache = {}
        self._elig_hash_cache = {}
        self._elig_feature_cache = {}

        # Full-brain lookup for DAN slots and MBON slots.
        dan_slot = np.full(brain.n, -1, np.int32)
        dan_slot[self.dan_ids] = np.arange(len(self.dan_ids), dtype=np.int32)
        self.dan_slot_gpu = self.xp.asarray(dan_slot)

        # Local DAN->MBON axis from the same CSR graph.  It is used both for
        # biological gates and to ensure engineering gates remain compartmental.
        axis = np.zeros((len(self.mbon_ids), len(self.dan_ids)), np.float32)
        dan_types = ct[self.dan_ids]
        valence = np.zeros(len(self.dan_ids), np.float32)
        valence[np.char.find(dan_types, "PAM") >= 0] = 1.0
        valence[np.char.find(dan_types, "PPL") >= 0] = -1.0
        valence[np.char.find(dan_types, "PPM") >= 0] = 0.5
        dan_set = set(int(n) for n in self.dan_ids)
        for j, post in enumerate(self.mbon_ids):
            lo, hi = int(indptr[post]), int(indptr[post + 1])
            for pos in range(lo, hi):
                pre = int(indices[pos])
                if pre in dan_set:
                    d = int(dan_slot[pre])
                    axis[j, d] = float(self.xp.asnumpy(csr.data[pos])) * valence[d]
            norm = np.abs(axis[j]).sum()
            if norm > 0:
                axis[j] /= norm
        self.dan_axis = self.xp.asarray(axis)
        local = axis.sum(axis=1)
        local_scale = float(np.max(np.abs(local))) if len(local) else 0.0
        if local_scale > 0:
            local = local / local_scale
        # MBONs without a direct annotated DAN edge do not receive a biological
        # gate; engineering mode likewise leaves those synapses untouched.
        self.engineering_axis = self.xp.asarray(local.astype(np.float32))
        # A fixed, connectome-derived sign lets the compact KC projection
        # retain the compartment/valence direction instead of summing PAM and
        # PPL writes to zero.  It is not an additional learned parameter.
        edge_sign = np.sign(local[self.edge_mbon_slot]).astype(np.float32)
        self.edge_sign_gpu = self.xp.asarray(edge_sign)
        self.last_gate = self.xp.zeros(len(self.mbon_ids), dtype=self.xp.float32)
        self._token_dt = float(brain.dt) * 6.0
        self._elig_decay = np.float32(np.exp(-self._token_dt / self.eligibility_tau))
        self._weight_decay = np.float32(np.exp(-self._token_dt / self.weight_tau))

    @property
    def n_edges(self) -> int:
        return int(len(self.edge_pos))

    def reset(self) -> None:
        """Clear learned modulation and restore the original KC->MBON weights."""
        current_pre = self.xp.asnumpy(
            self.brain._W.indices[self.edge_pos_gpu]).astype(np.int64, copy=False)
        if not np.array_equal(current_pre, self.edge_pre_ids):
            raise RuntimeError(
                "KC->MBON CSR edge identities changed after plasticity construction")
        self.modulation[...] = 0
        self.kc_eligibility[...] = 0
        self.brain._W.data[self.edge_pos_gpu] = self.w0_gpu

    def edge_identity_fingerprint(self) -> dict:
        """Return an audit-only fingerprint of cached KC presynaptic IDs."""
        x = np.asarray(self.edge_pre_ids, dtype=np.int64)
        return {
            'n': int(x.size),
            'sum': int(x.sum()) if x.size else 0,
            'sumsq': int(np.dot(x, x)) if x.size else 0,
            'first': int(x[0]) if x.size else -1,
            'last': int(x[-1]) if x.size else -1,
        }

    def decay_only(self, write_back: bool = True) -> None:
        """Apply passive local forgetting without a new KC×DAN write.

        This is used by controlled memory probes during silent gaps and
        readout pulses.  It keeps the biological slow state and its existing
        connectome edges intact, while preventing spontaneous DAN activity or
        the probe stimulus itself from creating a second memory write.
        """
        self.kc_eligibility *= self._elig_decay
        self.modulation *= self._weight_decay
        self.last_gate.fill(0)
        if write_back:
            self.brain._W.data[self.edge_pos_gpu] = (
                self.w0_gpu * (np.float32(1.0) + self.modulation))

    def _gate(self, dan_counts, teaching_signal: float | None):
        if self.mode == "engineering":
            if teaching_signal is None:
                raise ValueError("engineering mode requires teaching_signal")
            return self.engineering_axis * np.float32(teaching_signal)
        d = self.xp.asarray(dan_counts, dtype=self.xp.float32).reshape(-1)
        if len(d) != len(self.dan_ids):
            raise ValueError("dan_counts has the wrong length")
        gate = self.dan_axis @ d
        # Keep the local gate in a stable range across different DAN firing
        # rates, while preserving its sign and compartment pattern.
        scale = self.xp.max(self.xp.abs(gate))
        return gate / self.xp.maximum(scale, np.float32(1.0))

    def update(self, kc_counts, dan_counts=None, teaching_signal: float | None = None,
               write_back: bool = True) -> None:
        """Write one token's KC eligibility into existing KC->MBON edges.

        ``write_back`` is additive (E4B).  The default ``True`` path is
        byte-identical to the frozen E3 rule.  Passing ``False`` skips only the
        final CSR write-back line, so the KC/DAN coincidence update to
        ``modulation``/``kc_eligibility`` is unchanged while the MBON cannot
        read the store during the write phase.  This supports the required
        temporally restored write-phase-only KC->MBON lesion.
        """
        kc = self.xp.asarray(kc_counts, dtype=self.xp.float32).reshape(-1)
        if len(kc) != len(self.kc_ids):
            raise ValueError("kc_counts has the wrong length")
        self.kc_eligibility *= self._elig_decay
        # Normalise spike counts by the token's nominal six 20-ms integration
        # steps; this makes eta portable across k=4/6/8 calibration runs.
        kc_activity = kc / np.float32(6.0)
        self.kc_eligibility += kc_activity
        gate = self._gate(dan_counts, teaching_signal)
        self.last_gate = gate.copy()
        self.modulation *= self._weight_decay
        # Same-token KC/DAN coincidence is the canonical local learning
        # signal.  The eligibility trace is retained and reported as a
        # separate diagnostic, but feeding it through the persistent weight
        # update as well would double-smooth the signal and hide the recent
        # token under a long history.  Slow forgetting of ``modulation``
        # itself supplies the synaptic memory we measure here.
        # A dopamine-gated eligibility trace is the biologically plausible
        # delayed-credit variant.  At mix=0 this is the original same-token
        # coincidence rule; increasing the mix lets a later DAN event write
        # an earlier KC trace without adding any edge or global gradient.
        plastic_kc = ((np.float32(1.0 - self.eligibility_mix) * kc_activity) +
                      np.float32(self.eligibility_mix) * self.kc_eligibility)
        drive = gate[self.edge_mbon_slot_gpu] * plastic_kc[self.edge_kc_slot_gpu]
        self.modulation += np.float32(self.eta) * drive
        self.modulation = self.xp.clip(self.modulation,
                                       np.float32(-self.max_modulation),
                                       np.float32(self.max_modulation))
        if write_back:
            self.brain._W.data[self.edge_pos_gpu] = (
                self.w0_gpu * (np.float32(1.0) + self.modulation))

    def state_features(self):
        """Return compact KC and MBON weight-state features on the host.

        KC state is a valence-aligned signed sum of each KC's modulation over
        its existing MBON outputs; MBON state is the corresponding signed sum
        over incoming KCs.  Keeping the sign is important for the engineering
        ablation: opposite next-character write signals must remain
        distinguishable.  The alignment sign is derived once from the local
        DAN→MBON connectome, so it does not add a free global parameter.
        These are readouts of the dynamic synaptic state, not extra neural
        units.  A magnitude-only readout would turn ``+label`` and
        ``-label`` writes into the same feature and erase the very memory we
        are testing.
        """
        kc = self.xp.zeros(len(self.kc_ids), dtype=self.xp.float32)
        mbon = self.xp.zeros(len(self.mbon_ids), dtype=self.xp.float32)
        import cupyx
        cupyx.scatter_add(kc, self.edge_kc_slot_gpu,
                          self.modulation * self.edge_sign_gpu)
        cupyx.scatter_add(mbon, self.edge_mbon_slot_gpu, self.modulation)
        return self.xp.asnumpy(kc), self.xp.asnumpy(mbon)

    def edge_state_features(self, bins: int = 8192):
        """Return a fixed count-sketch of individual KC→MBON weights.

        Summing all outgoing edges per KC can cancel distinct MBON
        compartments.  This deterministic count-sketch keeps edge-level
        information while avoiding a 61k-dimensional dense host matrix.  The
        hash/sign assignment is fixed from connectome indices and is not
        trained, so it is a measurement compression rather than an extra
        model pathway.
        """
        bins = int(bins)
        if bins <= 0:
            raise ValueError("bins must be positive")
        import cupyx
        hs = self._edge_hash_cache.get(bins)
        if hs is None:
            h = self.xp.asarray((self.edge_pos * 2654435761) % bins,
                                dtype=self.xp.int32)
            s = self.xp.asarray(np.where((self.edge_pos * 2246822519) & 1,
                                         1.0, -1.0), dtype=self.xp.float32)
            self._edge_hash_cache[bins] = (h, s)
        else:
            h, s = hs
        out = self._edge_feature_cache.get(bins)
        if out is None:
            out = self.xp.zeros(bins, dtype=self.xp.float32)
            self._edge_feature_cache[bins] = out
        else:
            out.fill(0)
        cupyx.scatter_add(out, h, self.modulation * s)
        return self.xp.asnumpy(out)

    def eligibility_features(self):
        """Return the short-lived KC eligibility state separately.

        Eligibility is a biological pre-write state rather than a learned
        weight, so experiments should not silently fold it into
        ``state_features``.  Exposing it as an auxiliary channel lets us tell
        whether a failure is in the KC activity/credit-assignment path or in
        the persistent KC→MBON synapse itself.
        """
        kc = self.kc_eligibility.copy()
        mbon = self.xp.zeros(len(self.mbon_ids), dtype=self.xp.float32)
        import cupyx
        # Project KC eligibility through the existing KC→MBON edge incidence;
        # this is a fixed anatomical summary, not a new trainable pathway.
        cupyx.scatter_add(mbon, self.edge_mbon_slot_gpu,
                          self.kc_eligibility[self.edge_kc_slot_gpu])
        return self.xp.asnumpy(kc), self.xp.asnumpy(mbon)

    def eligibility_state_features(self, bins: int = 1024):
        """Fixed sketch of the KC eligibility state for a fast MB channel.

        This is intentionally separate from :meth:`edge_state_features`:
        eligibility is a short-lived pre-write trace, while edge modulation is
        persistent synaptic state.  Keeping both channels makes a fast/slow
        MB bank measurable without treating the transient trace as long-term
        memory.
        """
        bins = int(bins)
        if bins <= 0:
            raise ValueError("bins must be positive")
        import cupyx
        hs = self._elig_hash_cache.get(bins)
        if hs is None:
            slots = self.xp.arange(len(self.kc_ids), dtype=self.xp.int64)
            h = (slots * np.int64(2654435761)) % bins
            s = self.xp.asarray(np.where((np.arange(len(self.kc_ids), dtype=np.int64) *
                                          np.int64(2246822519)) & 1,
                                         1.0, -1.0), dtype=self.xp.float32)
            self._elig_hash_cache[bins] = (h.astype(self.xp.int32), s)
        else:
            h, s = hs
        out = self._elig_feature_cache.get(bins)
        if out is None:
            out = self.xp.zeros(bins, dtype=self.xp.float32)
            self._elig_feature_cache[bins] = out
        else:
            out.fill(0)
        cupyx.scatter_add(out, h, self.kc_eligibility * s)
        return self.xp.asnumpy(out)

    def restore(self) -> None:
        """Restore fixed weights when the experiment is finished."""
        self.brain._W.data[self.edge_pos_gpu] = self.w0_gpu
