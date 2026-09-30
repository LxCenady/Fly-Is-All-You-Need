# Metrics of the earlier language models (before the audit)

Copies of the metrics files written by the project's earlier training scripts (the harness in
`harness/lm/`). LM Fig. 1 and Fig. 2 read them. They are kept as historical records: the
models were trained on 2026-09-15..17 with the earlier protocol, and their training scripts
used absolute paths of that machine.

| File | Model | Written | Status |
|---|---|---|---|
| `exp7_metrics.json` | Phase 2 pilot, 2 steps per character (14.6%) | 2026-09-15 | historical |
| `flybrain_memoryrank512_skip3h32k_softmax.metrics.json` | earlier frozen reference: 42.55% / 3.298 BPC | 2026-09-16 | historical; its score is at n-gram level (LM Section V) |
| `plastic_memory_lm_biological_dual_probe_20k_l2p1.metrics.json` | best plastic model: 42.88% / 3.346 BPC | 2026-09-17 00:00 | **superseded**: produced before the CSR fix (2026-09-17 22:56), so its plastic writes hit the wrong synapses |

sha256:
- exp7_metrics.json `c0a114e76cd88fe64536e274a46875529c1c913f5ea00406b331c4500c4d4f63`
- flybrain_memoryrank512_skip3h32k_softmax.metrics.json `7c37f9d8862aafc660d7d77201965d876042f14405785366da93b05ef49b190d`
- plastic_memory_lm_biological_dual_probe_20k_l2p1.metrics.json `6a65864736faa99a3781993e17454b925a56ac38b63e4573a53ab883e2b8bd40`
