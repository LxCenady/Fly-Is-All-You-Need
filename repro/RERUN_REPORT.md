# Rerun report

The 17 headline experiments of the mechanism paper were rerun on 2026-09-30 with the commands in
`repro/headline.json` (`python repro/make.py headline OUTDIR`). The environment was:

| | |
|---|---|
| Commit | c376a24 |
| Simulator | flybrain 0.1.0.post1 (`brain.py` sha256 106e8dd2...) |
| Packages | CuPy 14.2.0, NumPy 2.5.3; lock sha256 ee611c4e... |
| GPU | RTX 3070 Ti Laptop, driver 591.59 |
| Data | `brain.npz` cc9bd1ec..., `weights.npz` c29919aa... |

Every output has a manifest beside it. They were compared number by number with the committed
files (`python repro/make.py compare OUTDIR`).

**Result: all 17 reproduce.**
- 8 are bit-identical (maximum difference 0).
- 8 differ only at the last float32 bit of membrane voltages (at most 2.9e-7), the known
  run-to-run nondeterminism of GPU sparse products.
- 1, `m3_kcmbon`, was replaced by its rerun (below).

Two corrections were needed on the way:

- **`m3_kcmbon.json` (M10) was replaced by the rerun.**
  - The committed file was written on 2026-09-28 at 17:46 by a revision of `m3_kcmbon.py`
    that was never committed; the script's first commit came later that day. With the committed
    code, the real-wiring rows reproduced bit-for-bit but the two KC->MBON shuffles did not.
    The shuffle draws had changed: item cosine 0.960-0.982 vs 0.961-0.982, cue specificity
    2.76-3.26 vs 2.81-3.25.
  - The current code is deterministic: two further runs gave identical results. The committed
    file is now the rerun, so committed code regenerates it.
  - The paper's statements from it hold: cosine 0.96-0.98 and unchanged cue specificity. The
    top-5 MBON share of the shuffles moved from 27-29 % to 26-29 %, and the paper was updated.
  - Fig. 5 was regenerated: the two shuffle points moved slightly.
- **The canonical command for `m4_curve_a192.json` needed its tau list.** The committed file
  used weight time constants 16 s and 1e12 s. `repro/headline.json` first omitted the argument,
  so the script's default of four values was used; the job now passes `16,1e12`.

Tolerance: |rerun - committed| <= 1e-05 + 0.001 x |committed| for every number.

| Job | Claims | Numbers compared | Verdict | Max abs. diff | Max rel. diff | Fields only in the rerun |
|---|---|---|---|---|---|---|
| m1_cross | M2, M3 | 1,888 | match | 2.53e-07 | 1.36e+00 | - |
| m1_cross_a192 | M2, M3 | 1,888 | match | 2.29e-07 | 4.43e+01 | - |
| m1_validate_g0 | M1 | 26 | match | 1.19e-07 | 5.88e-02 | - |
| m1_validate_g32 | M1 | 26 | match | 1.19e-07 | 1.00e+00 | - |
| m2_current_6pairs | M8 | 2,220 | match | 0.00e+00 | 0.00e+00 | - |
| m3_kcmbon | M10 | 87 | match | 0.00e+00 | 0.00e+00 | - |
| m4_curve_a192 | M4, M5 | 1,163 | match | 2.94e-07 | 1.82e+01 | - |
| m5_downstream | M19 | 808 | match | 2.14e-07 | 5.93e+00 | - |
| m5b_timing | M19 | 896 | match | 2.31e-07 | 1.23e+01 | - |
| m6_glom_route | M12 | 198 | match | 0.00e+00 | 0.00e+00 | - |
| m6_hemi | M11 | 1,344 | match | 0.00e+00 | 0.00e+00 | - |
| m6_side_a160 | M10 | 103 | match | 0.00e+00 | 0.00e+00 | input_code |
| m6_side_a192 | M10 | 103 | match | 0.00e+00 | 0.00e+00 | input_code |
| m7_map | M14 | 4,522 | match | 0.00e+00 | 0.00e+00 | - |
| m7_teacher | M16 | 32 | match | 0.00e+00 | 0.00e+00 | - |
| m8_interference | M18 | 124 | match | 0.00e+00 | 0.00e+00 | - |
| sweep_s1.5 | M6, M7 | 259 | match | 1.43e-07 | 3.02e+00 | - |
