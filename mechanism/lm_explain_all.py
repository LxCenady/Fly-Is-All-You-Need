"""Whole-brain comparison (CPU): KN-5 alone vs KN-5 + all brain features (clipped z),
and the hashed context head + all features, for the real and rewired connectomes.
Usage: lm_explain_all.py OUT.json"""
import json
import re
import sys

import numpy as np

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm_controls as K  # noqa: E402
import lm_explain as X  # noqa: E402

FEATS = ["kc", "mbon_v", "mbon_spk", "central"]
TAGS = {"real": ["night/s160_o0_c-1", "night/s160_o300000_c-1", "night/s160_o600000_c-1"],
        "class": ["conn/class_o0", "conn/class_o300000", "conn/class_o600000"],
        "full": ["conn/full_o0", "conn/full_o300000", "conn/full_o600000"]}


def main(out):
    res = {}
    for brain, tags in TAGS.items():
        for tag in tags:
            off = int(re.search(r"_o(\d+)", tag).group(1))
            ids = np.asarray([K.chars.index(c) for c in K.TEXT[off: off + 25001]], np.int64)
            z = np.load(X.E / (tag + ".npz"))
            Z = K.zclip(np.concatenate([z[f] for f in FEATS], 1), 0, 20000)
            kn5, y = X.kn_offsets(ids, 5)
            r = {"kn5": X.stack_eval(kn5, None, y), "kn5+all": X.stack_eval(kn5, Z, y)}
            x_ids, yy, cx = K.data(off, 25000)
            r["ctx+all"] = K.evaluate(np.concatenate([z[f] for f in FEATS], 1), yy, cx, 20000, 20000, 25000)
            res.setdefault(brain, []).append({"offset": off, **{k: (v if isinstance(v, dict) else
                                                                    {"acc": v[0], "bpc": v[1]}) for k, v in r.items()}})
            print(brain, off, {k: round((v["bpc"] if isinstance(v, dict) else v[1]), 4) for k, v in r.items()}, flush=True)
            json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
