"""Is APL (KC-sparsening inhibition) in the connectome, and how is PN->KC wired?"""
import numpy as np
from scipy import sparse

m = np.load(r"D:\flybrain_lm_cuda\data\brain.npz", allow_pickle=True)
ct = m["cell_type"].astype(str)
W = sparse.load_npz(r"D:\flybrain_lm_cuda\data\weights.npz").tocsr()   # rows = post
Wc = W.tocsc()
kc = np.char.find(ct, "KC") >= 0
pn = np.char.find(ct, "PN") >= 0
apl = np.flatnonzero(np.char.find(ct, "APL") >= 0)
print("APL ids", apl.tolist(), ct[apl].tolist())
for a in apl:
    col = Wc[:, a]
    tk = kc[col.indices]
    print(f"  APL {a} -> KC: {tk.sum()} edges, sum w {col.data[tk].sum():.4f}, neg frac {(col.data[tk] < 0).mean() if tk.any() else float('nan'):.2f}")
    row = W[a]
    fk = kc[row.indices]
    print(f"  KC -> APL {a}: {fk.sum()} edges, sum w {row.data[fk].sum():.4f}")
x = W[np.flatnonzero(kc)][:, np.flatnonzero(pn)]
nn = x.getnnz(axis=1)
print(f"PN->KC: KCs with PN input {int((nn > 0).sum())}/{int(kc.sum())}, PN partners per KC median {np.median(nn):.0f}, "
      f"mean summed PN weight per KC {np.asarray(x.sum(axis=1)).mean():.4f}")
neg = (W.data < 0).mean()
print(f"fraction of negative weights in whole W: {neg:.3f}")
