"""Input-output structure of individual KCs in the connectome (CPU only).

For every KC: input vector = PN->KC weights aggregated by PN type (180 types),
output vector = KC->MBON weights (97 MBONs).
1. Mantel-type test: correlation between KC-KC input similarity and KC-KC
   output similarity (cosine), over all KC pairs, against a null that permutes
   output profiles among KCs of the same annotated subtype (so subtype-level
   structure is preserved in the null and only finer matching is tested).
2. Same restricted to KC pairs within one subtype.
3. Clusters of KCs by input profile (k-means, k = 5, 20, 80, 300) for the
   cluster-constrained profile-swap experiment; saved to kc_input_clusters.npz.
4. How much of each KC's output is predictable from its input cluster:
   between-cluster share of output-profile variance, vs the subtype-shuffle null.
5. MBON names and in-degrees for reporting.
"""
import json
import sys

import numpy as np
from scipy import sparse

import paths  # noqa: E402  (mechanism/paths.py: data and output locations)
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(0)
m = np.load(paths.data_dir() / "brain.npz", allow_pickle=True)
ct = m["cell_type"].astype(str)
W = sparse.load_npz(paths.data_dir() / "weights.npz").tocsr()     # rows = post
sup = m["superclass"].astype(str)
W = (sparse.diags((np.char.find(sup, "sensory") < 0).astype(np.float32)) @ W).tocsr()
kc = np.flatnonzero(np.char.find(ct, "KC") >= 0)
mb = np.flatnonzero(np.char.find(ct, "MBON") >= 0)
pn = np.flatnonzero(np.char.find(ct, "PN") >= 0)
ptypes, pidx = np.unique(ct[pn], return_inverse=True)
T = sparse.csr_matrix((np.ones(len(pn)), (np.arange(len(pn)), pidx)), shape=(len(pn), len(ptypes)))
IN = (W[kc][:, pn] @ T).toarray()                    # KC x PN-type
OUT = W[mb][:, kc].T.toarray()                       # KC x MBON
keep = (IN.sum(1) > 0) & (np.abs(OUT).sum(1) > 0)
kc, IN, OUT = kc[keep], IN[keep], OUT[keep]
sub = ct[kc]
print(f"KCs with both PN input and MBON output: {len(kc)}")


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1)


Ui, Uo = unit(IN), unit(OUT)
S = rng.choice(len(kc), size=min(2000, len(kc)), replace=False)
iu = np.triu_indices(len(S), 1)
si = (Ui[S] @ Ui[S].T)[iu]


def mantel(Uo_, rows=S):
    so = (Uo_[rows] @ Uo_[rows].T)[iu]
    return float(np.corrcoef(si, so)[0, 1])


def within_subtype_perm(r):
    perm = np.arange(len(kc))
    for t in np.unique(sub):
        idx = np.flatnonzero(sub == t)
        perm[idx] = idx[r.permutation(len(idx))]
    return perm


real = mantel(Uo)
null = [mantel(Uo[within_subtype_perm(np.random.default_rng(i))]) for i in range(200)]
same = (sub[S][:, None] == sub[S][None, :])[iu]
so_real = (Uo[S] @ Uo[S].T)[iu]
r_within = float(np.corrcoef(si[same], so_real[same])[0, 1])
null_within = []
for i in range(200):
    p = within_subtype_perm(np.random.default_rng(1000 + i))
    so = (Uo[p][S] @ Uo[p][S].T)[iu]
    null_within.append(float(np.corrcoef(si[same], so[same])[0, 1]))

out = {"n_kc": int(len(kc)), "n_pn_types": int(len(ptypes)),
       "mantel_all_pairs": {"real": real, "null_mean": float(np.mean(null)),
                            "null_sd": float(np.std(null)), "null_max": float(np.max(null)),
                            "n_null": 200},
       "mantel_within_subtype_pairs": {"real": r_within, "null_mean": float(np.mean(null_within)),
                                       "null_sd": float(np.std(null_within)),
                                       "null_max": float(np.max(null_within))}}

# clusters on input profile and how much output variance they explain
clusters = {}
for k in (5, 20, 80, 300):
    _, lab = kmeans2(Ui, k, seed=np.random.default_rng(k), minit="++")
    clusters[k] = lab

    def between_share(O, lab=lab):
        mu = O.mean(0)
        tot = ((O - mu) ** 2).sum()
        bw = sum(((O[lab == c].mean(0) - mu) ** 2).sum() * (lab == c).sum()
                 for c in np.unique(lab))
        return float(bw / tot)

    nulls = [between_share(Uo[within_subtype_perm(np.random.default_rng(5000 + i))]) for i in range(50)]
    out[f"input_cluster_k{k}"] = {"output_between_share_real": between_share(Uo),
                                  "null_mean": float(np.mean(nulls)), "null_max": float(np.max(nulls)),
                                  "cluster_sizes_min_med_max": [int(np.bincount(lab).min()),
                                                                int(np.median(np.bincount(lab))),
                                                                int(np.bincount(lab).max())]}
np.savez(paths.out("kc_input_clusters.npz"), kc=kc,
         **{f"k{k}": v for k, v in clusters.items()}, subtype=sub)

# effective PN-type -> MBON map through KCs, and its specificity vs null
E = IN.T @ OUT                                            # PN-type x MBON
def spec(E_):
    U = unit(E_)
    G = U @ U.T
    return float(G[np.triu_indices(len(G), 1)].mean())
out["pntype_to_mbon_mean_cos"] = {"real": spec(E), "null_within_subtype": [
    spec(IN.T @ OUT[within_subtype_perm(np.random.default_rng(9000 + i))]) for i in range(20)]}
out["mbon_names"] = {str(i): str(ct[j]) for i, j in enumerate(mb)}
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "mbon_names"}, indent=1))
