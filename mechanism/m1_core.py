"""M1 core: complete-state snapshot / restore / transplant for the MB memory probe.

State inventory (everything that can carry history between tokens in the
canonical E4B/E5A single-lane protocol, noise_hz = 0):

fast state ``s`` (network dynamics + harness traces)
    brain.v          (n, 1) float32 membrane voltages
    brain.fired      flat int64 indices of last step's spikes (drives the next
                     step's synaptic input: the only "delay queue" in the model)
    brain.steps      int step counter (only used by refractory; refractory=0)
    trace_mbon / trace_central   harness spike traces (readout-side only)
    plastic.kc_eligibility       KC eligibility trace; with eligibility_mix=0
                                 it never enters an update, so it is inert
    plastic.last_gate            last DAN gate; diagnostic only, inert
slow state ``m`` (synaptic)
    plastic.modulation           61,210 float32, the learned KC->MBON state
    brain._W.data[edge_pos]      live weights = w0*(1+modulation); kept
                                 consistent with ``modulation`` on restore
not state (asserted constant)
    every other CSR entry; brain.rng (noise_hz=0 => draws never change v)

Nothing in lm/ is modified; the frozen probe module is imported.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

paths.use_harness()                       # harness lm/ modules + its sitecustomize (CuPy hook)
LM = paths.HARNESS / "lm"
import sitecustomize  # noqa: F401,E402
import mb_e5a_kinetics_v2 as v2  # noqa: E402
import mb_e4b_sequential as e4b  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402
from train_lm import load_corpus  # noqa: E402
from flybrain import FlyBrain  # noqa: E402

import os  # noqa: E402
from mb_plasticity import KCMBONPlasticity as _KP  # noqa: E402

# MB_SIGN=depress: biological sign convention.  Every DAN (PAM, PPL, PPM)
# depresses its compartment's KC->MBON synapses (Hige et al. 2015), instead of
# the hand-set PAM +1 / PPL -1 / PPM +0.5 valence.  Implemented by replacing
# the per-MBON DAN axis with -|axis| after construction; compartment pattern and
# normalisation are unchanged.
if os.environ.get("MB_SIGN") == "depress" and not getattr(_KP, "_sign_patched", False):
    _orig_init = _KP.__init__

    def _init_depress(self, *a, **kw):
        _orig_init(self, *a, **kw)
        self.dan_axis = -self.xp.abs(self.dan_axis)

    _KP.__init__ = _init_depress
    _KP._sign_patched = True

DATA = paths.data_dir()          # FLY_DATA (MB_DATA for rewired-connectome controls); see paths.py
CORPUS = paths.CORPUS
SETTLE = slice(6, 12)          # frozen settle window [6, 11] of the probe profile


def protocol_args(eta=0.02, weight_tau=16.0, noise_hz=0.0, seed=20260920):
    import argparse
    chars, _, _, _ = load_corpus(str(CORPUS), 128, 1, 1)
    chars = list(chars)
    ns = argparse.Namespace(**v2.REFERENCE_PROTOCOL)
    ns.corpus = str(CORPUS)
    ns.data = DATA
    ns.vocab = len(chars)
    ns.seed = int(seed)
    ns.repeats = 1
    ns.gap_tokens = 0
    ns.probe_sustain = None
    # Sensitivity overrides (unset = reference protocol): MB_GAIN, MB_TONIC, MB_NOISE (Hz).
    # With MB_NOISE > 0 the noise RNG is not part of the snapshot, so episodes see
    # independent noise realisations.
    if os.environ.get("MB_GAIN"):
        ns.gain = float(os.environ["MB_GAIN"])
    if os.environ.get("MB_TONIC"):
        ns.tonic = float(os.environ["MB_TONIC"])
    if os.environ.get("MB_NOISE"):
        noise_hz = float(os.environ["MB_NOISE"])
    ns.eta, ns.weight_tau, ns.noise_hz = float(eta), float(weight_tau), float(noise_hz)
    ns.noise_grid = [float(noise_hz)]
    ns.graded_tol, ns.graded_ulp = 1e-6, 16
    e4b._build_args(ns)
    ns.record_fingerprints = False
    return ns, chars


def build(args, condition="intact"):
    brain0 = FlyBrain(data=args.data, device="cuda", batch=1, seed=args.seed,
                      sensory_input=False)
    masks = pmp._masks(brain0)
    central = pmp._select_targets(brain0, args.target_selection, args.target_top)
    del brain0
    st = pmp._new_condition(args, condition, masks, args.vocab, central)
    p = st["plastic"]
    live = p.xp.asnumpy(st["brain"]._W.indices[p.edge_pos_gpu]).astype(np.int64)
    assert np.array_equal(live, p.edge_pre_ids), "KC->MBON identity drift"
    # The pre-fix bug struck at the first GPU sparse product (cuSPARSE re-sorted an unsorted CSR
    # in place), after the check above.  Do one product now and check again.
    _ = st["brain"].synaptic_input(st["brain"].fired)
    p.xp.cuda.Device().synchronize()
    live = p.xp.asnumpy(st["brain"]._W.indices[p.edge_pos_gpu]).astype(np.int64)
    assert np.array_equal(live, p.edge_pre_ids), "KC->MBON identity changed at the first SpMV (unfixed flybrain?)"
    assert bool(st["brain"]._W.has_sorted_indices), "GPU matrix not canonical (use flybrain 0.1.0.post1)"
    if os.environ.get("MB_CODE_SEED"):
        # Replication hook: a fresh random PN code for every character (new "stimuli");
        # unset = the reference codes (seed 3).  Glomerular codes: MB_GLOM_SEED (m3_pnkc).
        enc = st["enc"]
        st["enc"] = type(enc)(enc.channels, len(enc.codes), active=enc.active, drive=enc.drive,
                              window=enc.window, gamma=enc.gamma, seed=int(os.environ["MB_CODE_SEED"]))
    if os.environ.get("MB_LESION"):
        st["_lesion"] = _apl_lesion(st, os.environ["MB_LESION"])
    st["_other_mask"] = _other_positions(st)
    return st


def _apl_lesion(st, mode):
    """MB_LESION=apl_off: zero APL->KC edges.  MB_LESION=apl_fbX: scale KC->APL edges by X.
    CSR rows are postsynaptic, indices presynaptic (checked above for KC->MBON)."""
    m = np.load(DATA / "brain.npz", allow_pickle=True)
    ct = m["cell_type"].astype(str)
    is_kc = np.char.find(ct, "KC") >= 0
    is_apl = np.char.find(ct, "APL") >= 0
    W = st["brain"]._W
    xp = st["brain"].xp
    indptr = xp.asnumpy(W.indptr); indices = xp.asnumpy(W.indices)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    if mode == "apl_off":
        sel = is_kc[rows] & is_apl[indices]
        before = float(xp.asnumpy(W.data[xp.asarray(np.flatnonzero(sel))]).sum())
        W.data[xp.asarray(np.flatnonzero(sel))] = 0
    elif mode.startswith("apl_fb"):
        sel = is_apl[rows] & is_kc[indices]
        before = float(xp.asnumpy(W.data[xp.asarray(np.flatnonzero(sel))]).sum())
        W.data[xp.asarray(np.flatnonzero(sel))] *= float(mode[6:])
    else:
        raise ValueError(mode)
    assert not np.any(sel[xp.asnumpy(st["plastic"].edge_pos_gpu)]), "lesion touched KC->MBON"
    info = {"mode": mode, "n_apl": int(is_apl.sum()), "edges": int(sel.sum()), "weight_before": before}
    print("lesion", info, flush=True)
    return info


def _other_positions(st):
    xp = st["brain"].xp
    mask = xp.ones(st["brain"]._W.data.shape[0], dtype=bool)
    mask[st["plastic"].edge_pos_gpu] = False
    return mask


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def w_other_hash(st) -> str:
    b = st["brain"]
    return _h(b.xp.asnumpy(b._W.data[st["_other_mask"]]))


# --------------------------------------------------------------------------
# protocol pieces (all delegate to the frozen probe module)
# --------------------------------------------------------------------------
def reset(st, args):
    pmp._reset_episode(st, args)


def write_item(st, args, token_id, tokens=32, plastic=True):
    for _ in range(tokens):
        pmp._advance_token(st, args, token_id, allow_plastic=plastic,
                           pulse_dan=bool(args.dan_pulse > 0))


def gap(st, args, n):
    for _ in range(int(n)):
        pmp._advance_silent(st, args, allow_plastic=False)


def probe(st, args, token_id):
    """One probe token with the frozen drive / settle profile; no learning."""
    enc = st["enc"]
    old = enc.drive
    enc.drive = float(args.probe_drive)
    try:
        pmp._advance_token(st, args, int(token_id), allow_plastic=False,
                           capture_probe=False, record_profile=True,
                           settle_steps=int(args.probe_settle_steps))
    finally:
        enc.drive = old
    prof = st["_last_probe_profile"]
    return {"mbon": prof["v_pre_mbon"][SETTLE].astype(np.float64).ravel(),
            "central": prof["v_pre_central"][SETTLE].astype(np.float64).ravel(),
            "mbon_events": float(prof["mbon_events"].sum()),
            "mbon_raster": prof["mbon_events"].copy(),        # steps x 97 spike counts
            "central_events": float(prof["central_events"].sum())}


# --------------------------------------------------------------------------
# snapshot / restore
# --------------------------------------------------------------------------
FAST = ("v", "fired", "steps", "trace_mbon", "trace_central",
        "kc_eligibility", "last_gate")
SLOW = ("modulation", "w_slots")


def snapshot(st) -> dict:
    b, p = st["brain"], st["plastic"]
    return {
        "v": b.v.copy(), "fired": b.fired.copy(), "steps": int(b.steps),
        "trace_mbon": st["trace_mbon"].copy(),
        "trace_central": st["trace_central"].copy(),
        "kc_eligibility": p.kc_eligibility.copy(), "last_gate": p.last_gate.copy(),
        "modulation": p.modulation.copy(),
        "w_slots": b._W.data[p.edge_pos_gpu].copy(),
    }


def restore(st, fast: dict | None, slow: dict | None):
    """Install a fast state and/or a slow state.

    ``fast=None`` leaves the fast state as is; the special value ``"zero"``
    installs the silent reset state (v=0, no spikes, empty traces).
    ``slow="zero"`` installs modulation=0 and w=w0.
    """
    b, p = st["brain"], st["plastic"]
    xp = b.xp
    if isinstance(fast, str) and fast == "zero":
        b.v.fill(0)
        b.fired = xp.empty(0, xp.int64)
        st["trace_mbon"].fill(0)
        st["trace_central"].fill(0)
        p.kc_eligibility.fill(0)
        p.last_gate.fill(0)
    elif fast is not None:
        b.v[...] = fast["v"]
        b.fired = fast["fired"].copy()
        b.steps = int(fast["steps"])
        st["trace_mbon"][...] = fast["trace_mbon"]
        st["trace_central"][...] = fast["trace_central"]
        p.kc_eligibility[...] = fast["kc_eligibility"]
        p.last_gate[...] = fast["last_gate"]
    if isinstance(slow, str) and slow == "zero":
        p.modulation.fill(0)
        b._W.data[p.edge_pos_gpu] = p.w0_gpu
    elif slow is not None:
        p.modulation[...] = slow["modulation"]
        b._W.data[p.edge_pos_gpu] = slow["w_slots"]


def state_distance(a: dict, b: dict) -> dict:
    """Host-side distances between two snapshots, per component."""
    out = {}
    for k in ("v", "trace_mbon", "trace_central", "kc_eligibility", "modulation", "w_slots"):
        x = np.asarray(a[k].get(), np.float64).ravel()
        y = np.asarray(b[k].get(), np.float64).ravel()
        out[k] = {"l2": float(np.linalg.norm(x - y)),
                  "ref_l2": float(max(np.linalg.norm(x), np.linalg.norm(y)))}
    fa, fb = set(a["fired"].get().tolist()), set(b["fired"].get().tolist())
    out["fired"] = {"sym_diff": len(fa ^ fb), "n_a": len(fa), "n_b": len(fb)}
    return out


def snap_hash(s: dict) -> dict:
    return {k: (_h(s[k].get()) if hasattr(s[k], "get") else s[k]) for k in FAST + SLOW}
