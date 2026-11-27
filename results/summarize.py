"""Turn the per-epoch run records into the numbers the README and figures use.

  python results/summarize.py        -> writes results/summary.json and prints a table

Every run is scored the same way: the epoch with the lowest validation MSE is picked, and the test MSE
of that epoch is reported. "share" is the share of rating variance a model removes on the test set:
1 - MSE / MSE(global mean). The global mean scores 0%, a perfect model 100%.
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load(run):
    path = os.path.join(HERE, "runs", run, "epochs.jsonl")
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def pick(rows, max_epochs=None):
    rows = rows[:max_epochs] if max_epochs else rows
    best = min(rows, key=lambda r: r["val_mse"])
    last = rows[-1]
    return {"epochs_run": len(rows), "best_epoch": best["epoch"] + 1, "val": best["val_mse"], "test": best["test_mse"],
            "train_at_best": best.get("train_mse", best.get("train_loss")), "train_last": last.get("train_mse", last.get("train_loss")),
            "val_last": last["val_mse"]}


def probe(run):
    path = os.path.join(HERE, "runs", run, "probe.json")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    return json.loads(text[text.index("{"):]) if "{" in text else None


floor = json.load(open(os.path.join(HERE, "floor.json")))
VAR = floor["global_mean_test"]
share = lambda mse: 1.0 - mse / VAR

# The paper's own figures for this dataset (Table 3 and Table 7, Grocery & Gourmet Food).
PAPER = {"full": 0.747, "context": 0.774, "aspect": 0.778, "uiap": 0.795, "deepconn": 0.979, "carp": 0.752}

out = {"variance": VAR, "paper": PAPER, "paper_share": share(PAPER["full"]),
       "baselines": {"global_mean": {"test": VAR, "share": 0.0},
                     "bias": {"val": floor["bias_tuned"]["val"], "test": floor["bias_tuned"]["test"], "share": share(floor["bias_tuned"]["test"])},
                     "bias_mf_val_seeds": floor["mf_tuned"]["val_seeds"]},
       "single": {}, "groups": {}, "probes": {}}

SINGLE = {"as_shipped": "gpu_valsel_s1", "bracket_fix_only": "gpu_valsel_l2fix_s1", "both_fixes": "l2fix_fmfix_200", "equations_bias": "eq_bias_s1"}
for name, run in SINGLE.items():
    rows = load(run)
    if rows:
        out["single"][name] = dict(pick(rows), run=run)
        out["single"][name]["share"] = share(out["single"][name]["test"])
# the two released-graph runs at the same training length
rows = load("l2fix_fmfix_200")
if rows and "as_shipped" in out["single"]:
    n = out["single"]["as_shipped"]["epochs_run"]
    m = pick(rows, n)
    out["single"]["both_fixes_matched"] = dict(m, run="l2fix_fmfix_200", share=share(m["test"]), matched_epochs=n)

GROUPS = {"full": ["eq_s1", "full_s2", "full_s3"], "context": ["context_s%d" % s for s in (1, 2, 3)],
          "aspect": ["aspect_s%d" % s for s in (1, 2, 3)], "uiap": ["uiap_s%d" % s for s in (1, 2, 3)],
          "notarget": ["notarget_s%d" % s for s in (1, 2, 3)]}
for name, runs in GROUPS.items():
    got = [(r, pick(load(r), 40 if r == "eq_s1" else None)) for r in runs if load(r)]
    if not got:
        continue
    tests, vals = [g["test"] for _, g in got], [g["val"] for _, g in got]
    out["groups"][name] = {"runs": [r for r, _ in got], "n": len(got), "test": tests, "val": vals,
                           "test_mean": st.mean(tests), "test_min": min(tests), "test_max": max(tests),
                           "val_mean": st.mean(vals), "val_min": min(vals), "val_max": max(vals),
                           "best_epochs": [g["best_epoch"] for _, g in got], "train_at_best": [g["train_at_best"] for _, g in got],
                           "share_mean": share(st.mean(tests))}
full = out["groups"].get("full")
if full:
    for name in ("context", "aspect", "uiap", "notarget"):
        g = out["groups"].get(name)
        if g:  # relative change in test MSE against the full model, per seed and on the means
            g["rel_mean"] = g["test_mean"] / full["test_mean"] - 1.0
            g["rel_each"] = [t / full["test_mean"] - 1.0 for t in g["test"]]
    full["rel_each"] = [t / full["test_mean"] - 1.0 for t in full["test"]]
    for name in ("context", "aspect", "uiap"):
        out["paper"][name + "_rel"] = PAPER[name] / PAPER["full"] - 1.0

for name, run in list(SINGLE.items()) + [("equations", "eq_s1")]:
    p = probe(run)
    if p:
        out["probes"][name] = p

json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
pct = lambda x: "%5.1f%%" % (100 * x)
print("variance of test ratings (MSE of the global mean): %.4f" % VAR)
print("%-28s %8s %8s  %s" % ("", "test MSE", "share", "notes"))
print("%-28s %8.4f %8s" % ("user + item biases", out["baselines"]["bias"]["test"], pct(out["baselines"]["bias"]["share"])))
for k, v in out["single"].items():
    print("%-28s %8.4f %8s  best epoch %d of %d" % (k, v["test"], pct(v["share"]), v["best_epoch"], v["epochs_run"]))
for k, g in out["groups"].items():
    rel = ("  %+.1f%% vs full" % (100 * g["rel_mean"])) if "rel_mean" in g else ""
    print("%-28s %8.4f %8s  n=%d range %.4f-%.4f%s" % ("equations: " + k, g["test_mean"], pct(g["share_mean"]), g["n"], g["test_min"], g["test_max"], rel))
print("%-28s %8.4f %8s  as reported" % ("paper, RPMIA", PAPER["full"], pct(out["paper_share"])))
for k, p in out["probes"].items():
    print("probe %-20s real %.4f shuffled %.4f" % (k, p["test_mse_real_docs"], p["test_mse_shuffled_docs"]))
