"""Does CuPy's first SpMV canonicalize an unsorted CSR in place? (pre-fix brain.py path)"""
import hashlib, json, sys
import numpy as np
from scipy import sparse
import cupy as cp
from cupyx.scipy import sparse as cusparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mechanism"))
import paths  # noqa: E402
DATA = str(paths.data_dir())
meta = np.load(DATA + r"\brain.npz")
W = sparse.load_npz(DATA + r"\weights.npz")
sensory = np.char.find(meta["superclass"].astype(str), "sensory") >= 0
Wm = sparse.diags((~sensory).astype(np.float32)) @ W.tocsr()   # exactly the pre-fix line
G = cusparse.csr_matrix(Wm.tocsr().astype(np.float32))
h = lambda a: hashlib.sha256(cp.asnumpy(a).tobytes()).hexdigest()[:16]
before = {"flag": bool(G.has_sorted_indices), "idx": h(G.indices), "data": h(G.data)}
x = cp.zeros(G.shape[1], cp.float32); x[:1000] = 1
_ = G @ x
cp.cuda.Device().synchronize()
after = {"flag": bool(G.has_sorted_indices), "idx": h(G.indices), "data": h(G.data)}
S = Wm.tocsr().copy(); S.sum_duplicates(); S.sort_indices()
canon = {"idx": hashlib.sha256(S.indices.astype(np.int32).tobytes()).hexdigest()[:16]}
json.dump({"cupy": cp.__version__, "before": before, "after_first_spmv": after,
           "cpu_canonical_idx": canon, "indices_dtype": str(G.indices.dtype)},
          open(sys.argv[1], "w"), indent=2)
