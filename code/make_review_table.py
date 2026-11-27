#!/usr/bin/env python3
"""One row per review, with the repo's ids and the repo's 70/10/20 split.

This is the input of floor.py, the no-text baselines.
Columns: uid, iid, rating, time, split (train | val | test), summary, text.
The 28 raw reviews that are in none of the repo's split files are dropped.

Usage: make_review_table.py <dataset_G dir> <raw json.gz> <user2IdMapping.txt> <out.jsonl>
"""
import gzip
import json
import os
import sys

src, raw_path, umap_path, out_path = sys.argv[1:5]


def mapping(path):
    with open(path) as f:
        return {a: int(b) for a, b in (l.rstrip("\n").split("\t") for l in f if l.strip())}


item_id = mapping(os.path.join(src, "item2IdMapping.txt"))
user_id = mapping(umap_path)
split_of = {}
for name, tag in (("ValTrain", "train"), ("Validate", "val"), ("Test", "test")):
    with open(os.path.join(src, name + "Interaction.out")) as f:
        for line in f:
            if line.strip():
                u, i, r, t = line.split("\t")
                assert (int(u), int(i)) not in split_of
                split_of[(int(u), int(i))] = (tag, float(r), int(t))

counts = {"train": 0, "val": 0, "test": 0, "dropped": 0}
seen = set()
with gzip.open(raw_path, "rt", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    for line in f:
        r = json.loads(line)
        key = (user_id[r["reviewerID"]], item_id[r["asin"]])
        if key not in split_of or key in seen:
            counts["dropped"] += 1
            continue
        tag, rating, t = split_of[key]
        assert rating == float(r["overall"]) and t == int(r["unixReviewTime"]), key
        seen.add(key)
        counts[tag] += 1
        out.write(json.dumps({"uid": key[0], "iid": key[1], "rating": rating, "time": t, "split": tag,
                              "summary": r.get("summary", ""), "text": r.get("reviewText", "")}) + "\n")
assert len(seen) == len(split_of), (len(seen), len(split_of))
print("rows", counts)
