"""Is 'KC subtype' just 'which output channel'? Mean KC->MBON profile per subtype (x side),
cosine between subtype profiles, and within-subtype profile spread (CPU, connectome only)."""
import numpy as np
from scipy import sparse

m = np.load(r"D:\flybrain_lm_cuda\data\brain.npz", allow_pickle=True)
ct = m["cell_type"].astype(str)
W = sparse.load_npz(r"D:\flybrain_lm_cuda\data\weights.npz").tocsr()
kc = np.flatnonzero(np.char.find(ct, "KC") >= 0)
mb = np.flatnonzero(np.char.find(ct, "MBON") >= 0)
OUT = W[mb][:, kc].T.toarray()
keep = np.abs(OUT).sum(1) > 0
kc, OUT = kc[keep], OUT[keep]
sub = ct[kc]
side = m["side"].astype(str)[kc] if "side" in m.files else np.array(["?"] * len(kc))
print("fields:", [f for f in m.files][:30])
types = [t for t in np.unique(sub) if (sub == t).sum() >= 20]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


P = np.array([unit(OUT[sub == t].mean(0)) for t in types])
print("\nKC subtypes (n):", {t: int((sub == t).sum()) for t in types})
C = P @ P.T
print("\ncosine between subtype mean output profiles:")
print("          " + " ".join(f"{t[:8]:>8s}" for t in types))
for t, row in zip(types, C):
    print(f"{t[:9]:9s} " + " ".join(f"{v:8.2f}" for v in row))
U = unit(OUT)
print("\nwithin-subtype: mean cosine of single-KC profile to own subtype mean vs best other subtype mean")
for i, t in enumerate(types):
    s = U[sub == t] @ P.T
    own = s[:, i].mean(); other = np.delete(s, i, 1).max(1).mean()
    print(f"  {t:12s} own {own:.2f}  best-other {other:.2f}")
if side[0] != "?":
    print("\nsides:", dict(zip(*np.unique(side, return_counts=True))))
    for t in types[:6]:
        a = [unit(OUT[(sub == t) & (side == s)].mean(0)) for s in np.unique(side) if ((sub == t) & (side == s)).sum() > 5]
        if len(a) == 2:
            print(f"  {t:12s} cos(left mean, right mean) {float(a[0] @ a[1]):.2f}")
