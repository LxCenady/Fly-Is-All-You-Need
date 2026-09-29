"""Trained character GRU baseline on the same splits (CPU, torch).
1-layer GRU (embedding 32, hidden 256), truncated BPTT 100, batches of 32 random windows,
Adam 3e-3, early stopping on the last 10% of the training characters (checked every 100
steps, patience 8).  Validation BPC is computed by running the GRU over train+val with a
carried hidden state and scoring only the validation characters.
Usage: lm_gru.py OUT.json
"""
import json
import sys

import numpy as np
import torch

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm_controls as K  # noqa: E402

torch.set_num_threads(4)
SPLITS = [(0, 20000, 5000), (300000, 20000, 5000), (600000, 20000, 5000), (0, 100000, 20000)]


class GRU(torch.nn.Module):
    def __init__(self, V, e=32, h=256):
        super().__init__()
        self.emb = torch.nn.Embedding(V, e); self.rnn = torch.nn.GRU(e, h, batch_first=True)
        self.out = torch.nn.Linear(h, V)

    def forward(self, x, h=None):
        o, h = self.rnn(self.emb(x), h)
        return self.out(o), h


def bpc_over(model, ids, lo, hi):
    """Run over ids[0:hi] carrying state; mean BPC of predictions for targets ids[lo+1:hi+1]."""
    model.eval(); h = None; tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, hi, 1000):
            e = min(s + 1000, hi)
            x = torch.tensor(ids[s:e])[None]; yt = torch.tensor(ids[s + 1:e + 1])[None]
            lg, h = model(x, h)
            lp = torch.log_softmax(lg, -1).gather(-1, yt[..., None])[0, :, 0]
            m = torch.arange(s, e) >= lo
            tot += float(-lp[m].sum()); n += int(m.sum())
    model.train()
    return tot / n / np.log(2)


def run(off, ntr, nval, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ids = np.asarray([K.chars.index(c) for c in K.TEXT[off: off + ntr + nval + 1]], np.int64)
    cut = int(ntr * 0.9)
    model = GRU(K.V); opt = torch.optim.Adam(model.parameters(), 3e-3)
    best, best_state, bad, step = 1e9, None, 0, 0
    while bad < 8 and step < 20000:
        st = rng.integers(0, cut - 101, 32)
        x = torch.tensor(np.stack([ids[s:s + 100] for s in st]))
        y = torch.tensor(np.stack([ids[s + 1:s + 101] for s in st]))
        lg, _ = model(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, K.V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1
        if step % 100 == 0:
            hb = bpc_over(model, ids, cut, ntr)
            if hb < best - 1e-4:
                best, bad = hb, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
    model.load_state_dict(best_state)
    return {"offset": off, "train": ntr, "val": nval, "steps": step, "holdout_bpc": best,
            "val_bpc": bpc_over(model, ids, ntr, ntr + nval)}


if __name__ == "__main__":
    res = []
    for off, ntr, nval in SPLITS:
        r = run(off, ntr, nval); res.append(r)
        json.dump(res, open(sys.argv[1], "w"), indent=1)
        print(r, flush=True)
