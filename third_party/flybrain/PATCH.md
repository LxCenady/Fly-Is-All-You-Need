# flybrain 0.1.0.post1: what changed and why

**Base:** flybrain 0.1.0 as released on PyPI (2026-09-13). Every file except `flybrain/brain.py`
is byte-identical to the release: each was checked against the sha256 values in the wheel's
RECORD. The released `brain.py` was reconstructed exactly (sha256 `FC7P_9XTLjBlWCnjSziwQCTkm-rmYr0HgnnpYQx4mww`, 9,406 bytes,
matching RECORD). The only other change is the version string in `flybrain/__init__.py`.

**The bug.** With `sensory_input=False`, `FlyBrain.__init__` masks the rows of the weight matrix
with `sparse.diags(mask) @ W`. The result is a valid CSR matrix whose column indices are not
sorted within rows. On the GPU, CuPy/cuSPARSE may canonicalise such a matrix in place during
the first sparse product: it sorts the column indices and data within each row. Any code that
cached positions into `brain._W.indices` / `brain._W.data` before that first product (for
example, to apply plasticity to chosen synapses) then reads and writes the wrong synapses.
Nothing raises an error.

In the Fly-Is-All-You-Need project this moved 61,164 of 61,210 cached KC→MBON slots to
other synapses (audit: `audit_20260928/`). Frozen simulations are unaffected, because the
sparse product itself is the same.

**The fix.** Canonicalise on the CPU (`sum_duplicates()`, `sort_indices()`) before the matrix
is uploaded, so that the GPU copy is already canonical and is never re-sorted.

**Regression test.** `tests/test_csr_identity.py`:

1. On a small synthetic matrix, it shows that the pre-fix construction is re-sorted in place by
   the first GPU product in the installed CuPy.
2. It shows that the fixed `FlyBrain` keeps every edge where it was: `indptr`, `indices` and
   `data` are unchanged after the first steps.

```diff
--- flybrain-0.1.0/flybrain/brain.py
+++ flybrain-0.1.0.post1/flybrain/brain.py
@@ -107,6 +107,14 @@
                 raise RuntimeError("brain.npz has no superclass; run `flybrain build`")
             sensory = np.char.find(meta["superclass"].astype(str), "sensory") >= 0
             W = sparse.diags((~sensory).astype(np.float32)) @ W.tocsr()   # rows = postsynaptic
+        # The row mask above can leave a valid CSR with unsorted column
+        # indices.  CuPy/cuSPARSE is allowed to canonicalize such a matrix
+        # in-place on its first SpMV.  All mechanism probes cache anatomical
+        # edge offsets, so canonicalize *before* uploading/caching them; doing
+        # it lazily would silently move KC->MBON writes to different columns.
+        W = W.tocsr()
+        W.sum_duplicates()
+        W.sort_indices()
         if device == "cuda":
             import cupy
             from cupyx.scipy import sparse as cusparse
```
