# Sensitivity summary

## Output identity (item_cos_mean; lower = items more distinct)

| variant | code | real | side+subtype swap | subtype swap (sides mixed) | kc_perm | top5 real |
|---|---|---|---|---|---|---|
| base | 160 random | 0.756 | 0.745 | 0.942 | 0.964 | 0.573 |
| base | 192 glom | 0.885 | 0.873 | 0.902 | 0.981 | 0.480 |
| gain1.2 | 160 random | 0.709 | 0.708 | 0.901 | 0.853 | 0.572 |
| gain1.2 | 192 glom | 0.838 | 0.836 | 0.882 | 0.911 | 0.480 |
| gain1.8 | 160 random | 0.594 | 0.587 | 0.798 | 0.961 | 0.642 |
| gain1.8 | 192 glom | 0.654 | 0.643 | 0.667 | 0.841 | 0.461 |
| tonic0.03 | 160 random | 0.748 | 0.739 | 0.922 | 0.910 | 0.566 |
| tonic0.03 | 192 glom | 0.873 | 0.863 | 0.891 | 0.942 | 0.468 |
| tonic0.08 | 160 random | 0.636 | 0.620 | 0.896 | 0.979 | 0.666 |
| tonic0.08 | 192 glom | 0.724 | 0.709 | 0.724 | 0.927 | 0.457 |
| noise2 | 160 random | 0.733 | 0.723 | 0.940 | 0.982 | 0.591 |
| noise2 | 192 glom | 0.867 | 0.854 | 0.885 | 0.989 | 0.489 |
| noise5 | 160 random | 0.726 | 0.712 | 0.931 | 0.990 | 0.606 |
| noise5 | 192 glom | 0.813 | 0.801 | 0.809 | 0.963 | 0.446 |

## Retrieval key (M2, 6 pairs; cue probe = written item x)

| variant | point | cue specificity (median) | KC-identity scramble: magnitude kept | direction cos | content cos |
|---|---|---|---|---|---|
| base | 160x1.5 | 4.71 | 0.20 | 0.93 | 0.87 |
| base | 512x1.0 | 1.06 | 0.84 | 1.00 | 1.00 |
| gain1.2 | 160x1.5 | 9.89 | 0.08 | 0.70 | 0.52 |
| gain1.2 | 512x1.0 | 1.36 | 0.56 | 0.99 | 0.98 |
| gain1.8 | 160x1.5 | 2.78 | 0.69 | 0.98 | 0.97 |
| gain1.8 | 512x1.0 | 1.03 | 1.00 | 1.00 | 1.00 |
| tonic0.03 | 160x1.5 | 9.27 | 0.11 | 0.83 | 0.73 |
| tonic0.03 | 512x1.0 | 1.15 | 0.72 | 0.99 | 0.99 |
| tonic0.08 | 160x1.5 | 2.03 | 0.74 | 0.98 | 0.97 |
| tonic0.08 | 512x1.0 | 1.01 | 0.97 | 1.00 | 1.00 |
| noise2 | 160x1.5 | 2.94 | 0.38 | 0.98 | 0.93 |
| noise2 | 512x1.0 | 1.02 | 0.90 | 1.00 | 1.00 |
| noise5 | 160x1.5 | 1.95 | 0.67 | 0.99 | 0.96 |
| noise5 | 512x1.0 | 1.01 | 0.96 | 1.00 | 1.00 |