# Confirmatory run: results

Evaluated by repro/confirm_eval.py exactly as registered in repro/PREREGISTRATION.md.

- **H1: CONFIRMED.** cue specificity 160/192/512: 7.69 / 5.03 / 1.05 (need > 2, > 2, < 1.5); KC-identity shuffle keeps 160/192: 0.11 / 0.19 (need < 0.6).
- **H2: CONFIRMED.** side+subtype swap changes item cosine by -0.012 / -0.004 (need |.| < 0.03); KC->MBON shuffle by +0.163 / +0.202 (need > +0.10).
- **H3: CONFIRMED.** teacher share of output variance 0.89 (need > 0.60).
- **H4: CONFIRMED.** central changed in 1.00 of cells where the MBON spike count changed (need >= 0.90) and in 0.095 where it did not (need <= 0.10).
- **H5: CONFIRMED.** D_m / D_nat from 8 tokens: median 1.0000, minimum 0.9974 (need >= 0.99, >= 0.95).
  - lm_fly_c201: context 3.088, connectome 3.007, KN-5 2.612, class-rewired (matched) 3.006; memory k=2 0.85, k=4 0.35
  - lm_fly_c202: context 3.088, connectome 3.007, KN-5 2.612, class-rewired (matched) 3.009; memory k=2 0.82, k=4 0.31
  - lm_fly_c203: context 3.088, connectome 3.017, KN-5 2.612, class-rewired (matched) 3.004; memory k=2 0.85, k=4 0.35
  - lm_worm_c201: context 3.088, connectome 3.037, KN-5 2.612, class-rewired (matched) 3.057, fully rewired (matched) 3.121; memory k=2 0.64, k=4 0.21
  - lm_worm_c202: context 3.088, connectome 3.029, KN-5 2.612, class-rewired (matched) 3.063, fully rewired (matched) 3.032; memory k=2 0.68, k=4 0.23
  - lm_worm_c203: context 3.088, connectome 3.046, KN-5 2.612, class-rewired (matched) 3.082, fully rewired (matched) 3.090; memory k=2 0.68, k=4 0.22
- **H6: CONFIRMED.** fly, 3 fresh codes on unseen text: connectome below context in every case, KN-5 more than 0.3 below the connectome, memory k=2 in [0.60, 0.95] and k=4 < 0.45.
- **H7: CONFIRMED.** fly: class-rewired connectome at matched activity within 0.05 BPC of the real one in every case.
- **H8: CONFIRMED.** worm, 3 fresh codes on unseen text: connectome below context, KN-5 more than 0.3 below, class-rewired (matched) within 0.06, memory k=2 in [0.50, 0.90].

No hypothesis was re-tested with other seeds or thresholds after these results were seen.
