#!/usr/bin/env python3
"""Rebuild UserReviews.out / ItemReviews.out, which the RPMIA repo's loader needs but the repo omits.

Inputs : the repo's dataset_G folder (interaction splits, WordDict.out, item2IdMapping.txt)
         and the raw Amazon 2014 5-core file reviews_Grocery_and_Gourmet_Food_5.json.gz.
Outputs: two ready-to-train folders that share the same documents and differ only in the rating rows
  <out>/full/    trains on TrainInteraction (70% + the 10% validation rows), as the shipped script does
  <out>/valsel/  trains on ValTrainInteraction (70%), keeps validation aside to pick the epoch honestly
Documents are built from the 70% ValTrain reviews only. check_vocab.py shows that is what the authors
did: the words this tokeniser finds in those reviews are exactly the 19,987 keys of WordDict.out.

The repo ships no user-id mapping, so it is recovered by joining raw reviews to the interaction rows
on (item id, rating, unix time). The script stops if that join is not exact.

Usage: build_reviews.py <dataset_G dir> <raw json.gz> <out dir>
"""
import collections
import gzip
import json
import os
import re
import shutil
import sys

src, raw_path, out = sys.argv[1:4]
TOKEN = re.compile(r"[a-z0-9]+")
MAX_LEN = 300


def rows(name):
    with open(os.path.join(src, name)) as f:
        return [(int(a), int(b), float(c), int(d)) for a, b, c, d in (l.split("\t") for l in f if l.strip())]


item_id = {}
with open(os.path.join(src, "item2IdMapping.txt")) as f:
    for line in f:
        if line.strip():
            asin, i = line.rstrip("\n").split("\t")
            item_id[asin] = int(i)
vocab = set()
with open(os.path.join(src, "WordDict.out"), encoding="utf-8") as f:
    for line in f:
        if line.strip():
            vocab.add(line.split("\t")[0])

total = rows("TotalInteraction.out")
splits = {n: rows(n + "Interaction.out") for n in ("Train", "ValTrain", "Validate", "Test")}

raw = []
with gzip.open(raw_path, "rt", encoding="utf-8") as f:
    for line in f:
        raw.append(json.loads(line))
raw_users = {r["reviewerID"] for r in raw}
raw_items = {r["asin"] for r in raw}
print("raw: %d reviews, %d users, %d items" % (len(raw), len(raw_users), len(raw_items)))
print("repo: %d interactions, %d users, %d items" % (len(total), len({r[0] for r in total}), len({r[1] for r in total})))
missing = set(item_id) - raw_items
print("ASINs in item2IdMapping missing from raw: %d ; raw ASINs not in mapping: %d" % (len(missing), len(raw_items - set(item_id))))
assert not missing, "raw file does not cover the repo's items"

# --- recover the user-id mapping -------------------------------------------------------------
by_key = collections.defaultdict(set)
for r in raw:
    by_key[(item_id[r["asin"]], float(r["overall"]), int(r["unixReviewTime"]))].add(r["reviewerID"])
cands = {}
unmatched = 0
for u, i, rating, t in total:
    c = by_key.get((i, rating, t))
    if not c:
        unmatched += 1
        continue
    cands[u] = c if u not in cands else (cands[u] & c)
bad = {u: c for u, c in cands.items() if len(c) != 1}
print("interaction rows with no raw match: %d ; user ids not resolving to exactly one reviewer: %d" % (unmatched, len(bad)))
assert unmatched == 0 and not bad, "user-id join is not exact"
user_of = {next(iter(c)): u for u, c in cands.items()}
assert len(user_of) == len(cands), "two user ids map to one reviewer"
print("user mapping recovered: %d ids" % len(user_of))

pair_rows = collections.Counter((u, i) for u, i, _, _ in total)
print("duplicate (user,item) pairs in repo interactions: %d" % sum(1 for v in pair_rows.values() if v > 1))
raw_pairs = collections.Counter((user_of.get(r["reviewerID"]), item_id[r["asin"]]) for r in raw)
print("raw reviews whose (user,item) is not in repo interactions: %d" % sum(v for k, v in raw_pairs.items() if k not in pair_rows))

with open(os.path.join(out, "user2IdMapping.txt"), "w") as f:
    for rid, u in sorted(user_of.items(), key=lambda kv: kv[1]):
        f.write("%s\t%d\n" % (rid, u))

# --- tokenise once ---------------------------------------------------------------------------
n_tok = n_hit = 0
docs = []  # (user id, item id, in-vocab tokens), in raw-file order
for r in raw:
    toks = TOKEN.findall(r.get("reviewText", "").lower())
    kept = [w for w in toks if w in vocab]
    n_tok += len(toks)
    n_hit += len(kept)
    docs.append((user_of[r["reviewerID"]], item_id[r["asin"]], kept))
print("tokens: %d, in vocabulary: %d (%.1f%%); kept words per review: %.2f" % (n_tok, n_hit, 100.0 * n_hit / n_tok, n_hit / len(docs)))

n_users = 1 + max(r[0] for r in total)
n_items = 1 + max(r[1] for r in total)


def build(name, doc_split, copies):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    allowed = {(u, i) for u, i, _, _ in splits[doc_split]}
    udoc = collections.defaultdict(list)
    idoc = collections.defaultdict(list)
    used = 0
    for u, i, toks in docs:
        if (u, i) in allowed:
            used += 1
            udoc[u].extend(toks)
            idoc[i].extend(toks)
    for fname, n, table in (("UserReviews.out", n_users, udoc), ("ItemReviews.out", n_items, idoc)):
        with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
            for k in range(n):  # every id gets a line; the loader sizes the rating matrix from these
                f.write("%d\t%s\n" % (k, " ".join(table.get(k, []))))
        lens = [min(len(table.get(k, [])), MAX_LEN) for k in range(n)]
        full = [len(table.get(k, [])) for k in range(n)]
        print("  %s/%s: %d lines, empty %d, mean words %.1f (untruncated %.1f), at cap %d"
              % (name, fname, n, sum(1 for x in full if x == 0), sum(lens) / n, sum(full) / n, sum(1 for x in full if x >= MAX_LEN)))
    # one line per training review, in the order the documents were concatenated: lets a model rebuild a
    # document with one review left out (RPMIA_paper.py, RPMIA_EXCLUDE_TARGET)
    with open(os.path.join(d, "ReviewTokens.out"), "w", encoding="utf-8") as f:
        for u, i, toks in docs:
            if (u, i) in allowed:
                f.write("%d\t%d\t%s\n" % (u, i, " ".join(toks)))
    shutil.copy(os.path.join(src, "WordDict.out"), os.path.join(d, "WordDict.out"))
    for src_name, dst_name in copies:
        shutil.copy(os.path.join(src, src_name), os.path.join(d, dst_name))
    print("  %s: documents built from %d reviews in %s" % (name, used, doc_split))


build("full", "ValTrain", [("TrainInteraction.out", "TrainInteraction.out"), ("TestInteraction.out", "TestInteraction.out")])
build("valsel", "ValTrain", [("ValTrainInteraction.out", "TrainInteraction.out"), ("TestInteraction.out", "TestInteraction.out"),
                             ("ValidateInteraction.out", "ValidateInteraction.out")])
print("paper Table 2, Grocery: 57.34 words per review, 177.24 per user, 155 per item")
print("BUILD_OK")
