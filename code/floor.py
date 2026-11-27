#!/usr/bin/env python3
"""The no-text floor: what ratings alone give on the repo's split.

Step 1 re-derives the earlier hand-rolled number (damped user and item biases, damping 5, clipped to
1..5: test MSE 0.9742) to prove this loader and metric agree with it.
Step 2 tunes on VALIDATION only: bias damping, then biased matrix factorisation (rank, weight decay,
epochs). Test is read once, for the configuration validation picked, and reported next to step 1.
Step 3 writes the chosen model's predictions for every row (out-of-fold for train rows, 5 folds).
Nothing in this repository uses that file.

Usage: floor.py <reviews.jsonl> <out dir>
"""
import json
import os
import sys

import numpy as np
import torch

path, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
U = 1 + max(r["uid"] for r in rows)
I = 1 + max(r["iid"] for r in rows)


def arr(split):
    sel = [r for r in rows if r["split"] == split]
    return (np.array([r["uid"] for r in sel]), np.array([r["iid"] for r in sel]), np.array([r["rating"] for r in sel], dtype=np.float64))


tr, va, te = arr("train"), arr("val"), arr("test")
print("users %d items %d | train %d val %d test %d" % (U, I, len(tr[2]), len(va[2]), len(te[2])))


def mse(pred, y, clip=True):
    if clip:
        pred = np.clip(pred, 1.0, 5.0)
    return float(np.mean((pred - y) ** 2))


def fit_bias(train, damp_u, damp_i, iters=20):
    u, i, y = train
    mu = y.mean()
    bu, bi = np.zeros(U), np.zeros(I)
    for _ in range(iters):
        bi = np.bincount(i, weights=y - mu - bu[u], minlength=I) / (damp_i + np.bincount(i, minlength=I))
        bu = np.bincount(u, weights=y - mu - bi[i], minlength=U) / (damp_u + np.bincount(u, minlength=U))
    return mu, bu, bi


def bias_pred(model, data):
    mu, bu, bi = model
    return mu + bu[data[0]] + bi[data[1]]


results = {}
# ---- step 1: the earlier number ---------------------------------------------------------------
m5 = fit_bias(tr, 5.0, 5.0, iters=15)
results["global_mean_test"] = mse(np.full(len(te[2]), tr[2].mean()), te[2])
results["bias_damp5_test"] = mse(bias_pred(m5, te), te[2])
print("global mean: test %.4f | biases damping 5: test %.4f (earlier hand-rolled: 0.9742)" % (results["global_mean_test"], results["bias_damp5_test"]))

# ---- step 2a: bias damping, on validation -------------------------------------------------------
grid = [1, 2, 3, 5, 8, 12, 20, 30]
best = min(((mse(bias_pred(fit_bias(tr, a, b), va), va[2]), a, b) for a in grid for b in grid))
print("biases, tuned damping: val %.4f at user %g, item %g" % best)
results["bias_tuned"] = {"val": best[0], "damp_user": best[1], "damp_item": best[2]}
bias_model = fit_bias(tr, best[1], best[2])

# ---- step 2b: biased MF on top of the tuned biases ----------------------------------------------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_mf(train, base_model, rank, wd, epochs, seed, evals=()):
    """Learns p_u . q_i on the residual of the bias model. Returns the factor matrices and a per-epoch MSE trace on each
    eval set."""
    torch.manual_seed(seed)
    u = torch.tensor(train[0], device=dev)
    i = torch.tensor(train[1], device=dev)
    y = torch.tensor(train[2] - bias_pred(base_model, train), dtype=torch.float32, device=dev)
    P = torch.nn.Parameter(0.05 * torch.randn(U, rank, device=dev))
    Q = torch.nn.Parameter(0.05 * torch.randn(I, rank, device=dev))
    opt = torch.optim.Adam([P, Q], lr=5e-3)
    trace = []
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 2048):
            b = perm[s:s + 2048]
            pu, qi = P[u[b]], Q[i[b]]
            loss = ((pu * qi).sum(1) - y[b]).pow(2).mean() + wd * (pu.pow(2).sum(1) + qi.pow(2).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            Pn, Qn = P.cpu().numpy().astype(np.float64), Q.cpu().numpy().astype(np.float64)
            trace.append([mse(bias_pred(base_model, d) + (Pn[d[0]] * Qn[d[1]]).sum(1), d[2]) for d in evals])
    return (Pn, Qn), trace


mf_best = None
for rank in (4, 8, 16, 32):
    for wd in (0.02, 0.05, 0.1, 0.2, 0.4):
        _, trace = fit_mf(tr, bias_model, rank, wd, 40, 1, evals=(va,))
        ep = int(np.argmin([t[0] for t in trace]))
        print("  mf rank %2d wd %.2f: best val %.4f at epoch %d" % (rank, wd, trace[ep][0], ep + 1), flush=True)
        if mf_best is None or trace[ep][0] < mf_best[0]:
            mf_best = (trace[ep][0], rank, wd, ep + 1)
print("biased MF, tuned: val %.4f at rank %d, wd %g, epochs %d" % mf_best)
results["mf_tuned"] = {"val": mf_best[0], "rank": mf_best[1], "wd": mf_best[2], "epochs": mf_best[3]}

# seed spread of the chosen MF on validation
vals = []
for seed in (1, 2, 3):
    _, trace = fit_mf(tr, bias_model, mf_best[1], mf_best[2], mf_best[3], seed, evals=(va,))
    vals.append(trace[-1][0])
results["mf_tuned"]["val_seeds"] = vals
print("  chosen MF over 3 seeds: val %s" % ", ".join("%.4f" % v for v in vals))

use_mf = float(np.mean(vals)) < results["bias_tuned"]["val"] - 0.001
results["floor_model"] = "bias+mf" if use_mf else "bias"
print("floor model: %s" % results["floor_model"])


def floor_fit_predict(train, targets, seed=1):
    bm = fit_bias(train, best[1], best[2])
    preds = [bias_pred(bm, d) for d in targets]
    if use_mf:
        (P, Q), _ = fit_mf(train, bm, mf_best[1], mf_best[2], mf_best[3], seed)
        preds = [p + (P[d[0]] * Q[d[1]]).sum(1) for p, d in zip(preds, targets)]
    return preds


# ---- test, once, for the chosen floor ------------------------------------------------------------
p_va, p_te = floor_fit_predict(tr, [va, te])
results["floor_val"] = mse(p_va, va[2])
results["floor_test"] = mse(p_te, te[2])
results["bias_tuned"]["test"] = mse(bias_pred(bias_model, te), te[2])
print("FLOOR %s: val %.4f test %.4f | tuned biases alone: test %.4f" % (results["floor_model"], results["floor_val"], results["floor_test"], results["bias_tuned"]["test"]))

# ---- step 3: predictions for every row (train rows out-of-fold) -----------------------------------
rng = np.random.RandomState(1)
fold = rng.randint(0, 5, len(tr[2]))
oof = np.zeros(len(tr[2]))
for k in range(5):
    inside = fold != k
    sub = tuple(a[inside] for a in tr)
    held = tuple(a[~inside] for a in tr)
    oof[~inside] = floor_fit_predict(sub, [held])[0]
results["floor_train_oof"] = mse(oof, tr[2])
print("train out-of-fold floor MSE %.4f" % results["floor_train_oof"])
lookup = {}
for data, pred in ((tr, oof), (va, p_va), (te, p_te)):
    for u, i, p in zip(data[0], data[1], pred):
        lookup[(int(u), int(i))] = float(p)
with open(os.path.join(out_dir, "floor_pred.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps({"uid": r["uid"], "iid": r["iid"], "floor": lookup[(r["uid"], r["iid"])]}) + "\n")
with open(os.path.join(out_dir, "floor.json"), "w") as f:
    json.dump(results, f, indent=1)
print("FLOOR_DONE")
