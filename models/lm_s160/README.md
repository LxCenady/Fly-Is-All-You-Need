# Revised connectome-reservoir character LM (sparse 160, frozen brain)

Readout weights of the revised language model described in `paper_lm/`.

| | |
|---|---|
| Brain | MaleCNS v1.0, flybrain LIF, protocol of `mechanism/m1_core.py`; frozen (no plasticity) |
| Input | random PN symbol codes, 160 active PNs, drive x1.5 (sparse KC regime), default codes |
| Features | KC spike counts (4064), MBON voltage (97), MBON spikes (97), 256 central neurons trace + voltage (512) = 4770 |
| Readout | softmax: `b + E[ctx] + ((x - mu) / sd) @ W`, then divide by temperature |
| Context | hashed order-3 character context, 32768 buckets (`lm_mech.ctx_ids`) |
| Training | TinyShakespeare chars [0, 20000); L2 1e-2 fixed in advance; 15 epochs Adam |
| Validation | chars [20000, 25000): **42.0% accuracy, 3.180 BPC** (context head alone: 42.9% / 3.205) |

Files: `readout.npz` (W, E as float16, b, mu, sd) and `model.json` (metrics, vocabulary, config).
Produced by `mechanism/export_lm_model.py`.

**Caveats.** The readout is useless without the brain simulator: features for new
text must be computed by running the connectome (`mechanism/lm_mech.py`), which
needs the flybrain package and the MaleCNS data. Readout fitting is noisy: refitting
on the same features with other mini-batch orders gives 3.120–3.180 BPC, so this
model's advantage over the context head alone is small. Across 3 text segments x
3 code draws the sparse brain lowers BPC by ~0.08 at this data size, but with
100k training characters the gain falls to ~0.01 (see `paper_lm/`).
