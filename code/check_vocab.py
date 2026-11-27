#!/usr/bin/env python3
"""Which reviews did the authors build their vocabulary from? Compare WordDict.out with the
retainWordVoca.out words that this tokeniser finds in each candidate set of reviews.
Usage: check_vocab.py <dataset_G dir> <raw json.gz> <user2IdMapping.txt>"""
import gzip, json, os, re, sys
src, raw_path, umap_path = sys.argv[1:4]
TOKEN = re.compile(r"[a-z0-9]+")
word_dict = {l.split("\t")[0] for l in open(os.path.join(src, "WordDict.out"), encoding="utf-8") if l.strip()}
retain = {l.strip() for l in open(os.path.join(src, "retainWordVoca.out"), encoding="utf-8") if l.strip()}
item_id = dict((a, int(b)) for a, b in (l.rstrip("\n").split("\t") for l in open(os.path.join(src, "item2IdMapping.txt")) if l.strip()))
user_id = dict((a, int(b)) for a, b in (l.rstrip("\n").split("\t") for l in open(umap_path) if l.strip()))
pairs = {}
for name in ("Train", "ValTrain", "Total"):
    pairs[name] = {(int(a), int(b)) for a, b, _, _ in (l.split("\t") for l in open(os.path.join(src, name + "Interaction.out")) if l.strip())}
seen = {n: set() for n in pairs}
seen["AllRaw"] = set()
for line in gzip.open(raw_path, "rt", encoding="utf-8"):
    r = json.loads(line)
    toks = set(TOKEN.findall(r.get("reviewText", "").lower())) & retain
    key = (user_id[r["reviewerID"]], item_id[r["asin"]])
    seen["AllRaw"] |= toks
    for n, p in pairs.items():
        if key in p:
            seen[n] |= toks
print("WordDict %d words, retainWordVoca %d words, WordDict - retain: %d" % (len(word_dict), len(retain), len(word_dict - retain)))
for n, s in seen.items():
    print("%-8s retain words found: %5d | in WordDict but not found: %4d | found but not in WordDict: %4d | exact match: %s"
          % (n, len(s), len(word_dict - s), len(s - word_dict), s == word_dict))
