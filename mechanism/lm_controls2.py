"""Follow-up controls (CPU): does the brain add anything beyond explicit recent history?
Sets (all with the context head, z clipped at +-3, 3 segments, 20k/5k):
  onehot4, onehot8           explicit one-hot of the last 4 / 8 characters
  brain_kc+onehot4           KC counts together with onehot4
  randexp at L2 3e-2, 1e-1, 3e-1   the random expansion with stronger regularisation
Usage: lm_controls2.py OUT.json
"""
import json
import os
import sys

import numpy as np

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm_controls as K  # noqa: E402


def main(out):
    res = {}
    for off in (0, 300000, 600000):
        x_ids, y, cx = K.data(off, 25000)
        kc = np.load(os.path.join(K.NIGHT, f"s160_o{off}_c-1.npz"))["kc"]
        oh4 = K.onehot_lags(x_ids, 4)
        rx = K.randexp(x_ids)
        sets = [("onehot4", oh4, 5e-3), ("onehot8", K.onehot_lags(x_ids, 8), 5e-3),
                ("brain_kc+onehot4", np.concatenate([kc, oh4], 1), 5e-3),
                ("randexp_l2_3e-2", rx, 3e-2), ("randexp_l2_1e-1", rx, 1e-1), ("randexp_l2_3e-1", rx, 3e-1)]
        for name, X, l2 in sets:
            K.L2 = l2
            r = K.evaluate(X, y, cx, 20000, 20000, 25000)
            res.setdefault(name, []).append({"offset": off, "l2": l2, **r})
            open(out, "w").write(json.dumps(res, indent=1))
            print(f"{off} {name} {r['acc']:.4f} {r['bpc']:.4f}", flush=True)
        K.L2 = 5e-3


if __name__ == "__main__":
    main(sys.argv[1])
