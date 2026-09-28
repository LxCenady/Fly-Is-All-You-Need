"""Does APL fire during the standard protocol?  Count APL spikes per token."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402
from batchroll import _cuda_step_no_host_copy  # noqa: E402,F401

args, chars = C.protocol_args()
st = C.build(args)
b = st["brain"]
ct = np.asarray(b.cell_type).astype(str)
apl = np.flatnonzero(ct == "APL")
orig = pmp._cuda_step_no_host_copy
counts = {"n": 0}


def counting_step(brain):
    fired = orig(brain)
    f = fired.get()
    counts["n"] += int(np.isin(f, apl).sum())
    return fired


pmp._cuda_step_no_host_copy = counting_step
C.reset(st, args)
tid = chars.index("a")
per = []
for t in range(32):
    counts["n"] = 0
    pmp._advance_token(st, args, tid, allow_plastic=False, pulse_dan=True)
    per.append(counts["n"])
counts["n"] = 0
C.gap(st, args, 32)
gap_n = counts["n"]
print("APL ids", apl.tolist())
print("APL spikes per write token (32 tokens x 6 steps):", per)
print("APL spikes during 32-token gap:", gap_n)
print("APL max voltage now:", float(b.v[b.xp.asarray(apl), 0].max().get()))
