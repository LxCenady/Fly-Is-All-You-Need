# Overnight summary

queue status at 2026-09-29T02:47:51: running ['lm2_s160_100k', 'lm2_d512_100k'], 3 remaining

## 1. Fixed config (all features + context, L2 1e-2), 20k/5k

| code | runs | val BPC | val acc | train acc |
|---|---|---|---|---|
| s160 | 9/9 | 3.039 ± 0.089 | 0.444 ± 0.015 | 0.598 ± 0.007 |
| s192 | 9/9 | 3.107 ± 0.096 | 0.435 ± 0.014 | 0.597 ± 0.007 |
| d512 | 9/9 | 3.542 ± 0.276 | 0.367 ± 0.032 | 0.744 ± 0.083 |

missing: none

## 2. Holdout-selected (lm_select), paired against context only

### select_s160 (9 caches)

| segment | seed | selected | brain+ctx BPC | ctx-only BPC | brain-only BPC | delta |
|---|---|---|---|---|---|---|
| 0 | -1 | kc, 0.01 | 3.144 | 3.205 | 3.244 | -0.061 |
| 0 | 11 | nokc, 0.005 | 3.092 | 3.205 | 3.507 | -0.113 |
| 0 | 23 | nokc, 0.005 | 3.104 | 3.205 | 3.534 | -0.101 |
| 300000 | -1 | nokc, 0.005 | 3.051 | 3.115 | 3.653 | -0.063 |
| 300000 | 11 | nokc, 0.005 | 3.045 | 3.115 | 3.624 | -0.070 |
| 300000 | 23 | nokc, 0.01 | 3.060 | 3.115 | 3.717 | -0.054 |
| 600000 | -1 | nokc, 0.005 | 2.895 | 2.971 | 3.431 | -0.076 |
| 600000 | 11 | nokc, 0.005 | 2.895 | 2.971 | 3.391 | -0.076 |
| 600000 | 23 | nokc, 0.005 | 2.881 | 2.971 | 3.398 | -0.090 |

delta mean -0.078 ± 0.020; brain helps in 9/9 caches

### select_s192 (9 caches)

| segment | seed | selected | brain+ctx BPC | ctx-only BPC | brain-only BPC | delta |
|---|---|---|---|---|---|---|
| 0 | -1 | nokc, 0.005 | 3.124 | 3.205 | 3.524 | -0.081 |
| 0 | 11 | nokc, 0.005 | 3.118 | 3.205 | 3.523 | -0.087 |
| 0 | 23 | nokc, 0.005 | 3.100 | 3.205 | 3.474 | -0.105 |
| 300000 | -1 | nokc, 0.005 | 3.046 | 3.115 | 3.606 | -0.069 |
| 300000 | 11 | nokc, 0.005 | 3.055 | 3.115 | 3.626 | -0.060 |
| 300000 | 23 | nokc, 0.005 | 3.051 | 3.115 | 3.607 | -0.064 |
| 600000 | -1 | nokc, 0.005 | 2.890 | 2.971 | 3.382 | -0.081 |
| 600000 | 11 | nokc, 0.005 | 2.898 | 2.971 | 3.396 | -0.073 |
| 600000 | 23 | nokc, 0.005 | 2.907 | 2.971 | 3.410 | -0.064 |

delta mean -0.076 ± 0.014; brain helps in 9/9 caches

### select_d512 (9 caches)

| segment | seed | selected | brain+ctx BPC | ctx-only BPC | brain-only BPC | delta |
|---|---|---|---|---|---|---|
| 0 | -1 | nokc, 0.03 | 3.192 | 3.205 | 3.850 | -0.013 |
| 0 | 11 | nokc, 0.01 | 3.182 | 3.205 | 3.739 | -0.023 |
| 0 | 23 | nokc, 0.03 | 3.193 | 3.205 | 3.857 | -0.012 |
| 300000 | -1 | nokc, 0.01 | 3.093 | 3.115 | 3.859 | -0.022 |
| 300000 | 11 | nokc, 0.01 | 3.080 | 3.115 | 3.824 | -0.035 |
| 300000 | 23 | nokc, 0.03 | 3.133 | 3.115 | 4.032 | +0.018 |
| 600000 | -1 | nokc, 0.03 | 2.980 | 2.971 | 3.823 | +0.010 |
| 600000 | 11 | nokc, 0.03 | 2.982 | 2.971 | 3.824 | +0.011 |
| 600000 | 23 | nokc, 0.03 | 2.978 | 2.971 | 3.809 | +0.007 |

delta mean -0.007 ± 0.019; brain helps in 5/9 caches

### select_s160_100k (1 caches)

| segment | seed | selected | brain+ctx BPC | ctx-only BPC | brain-only BPC | delta |
|---|---|---|---|---|---|---|
| 0 | s160_100k | nokc, 0.005 | 2.698 | 2.682 | 3.510 | +0.016 |

delta mean 0.016; brain helps in 0/1 caches

### select_d512_100k (1 caches)

| segment | seed | selected | brain+ctx BPC | ctx-only BPC | brain-only BPC | delta |
|---|---|---|---|---|---|---|
| 0 | d512_100k | nokc, 0.03 | 2.726 | 2.682 | 3.873 | +0.044 |

delta mean 0.044; brain helps in 0/1 caches

## 3. Mechanism jobs

- mech_route_160_random: present; keys list[9]
- mech_route_160_glom: present; keys list[9]
- mech_route_192_random: present; keys list[9]
- mech_route_192_glom: present; keys list[9]
- mech_m2_12pairs: present; keys list[3]
