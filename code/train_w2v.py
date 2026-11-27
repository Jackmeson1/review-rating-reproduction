#!/usr/bin/env python3
"""Train the 50-d Word2Vec vectors the paper's Eq.1 calls for.

The paper says "a pretrained Word2Vec model" with 50-d vectors and gives no source. FILL: skip-gram
Word2Vec (gensim defaults otherwise: window 5, negative 5, 5 epochs) trained on the training-split user
documents, so no validation or test review is seen. One worker and a fixed seed keep it repeatable.

Output format is the one the released word2vec_word_embed() reads: word<TAB>v1 v2 ... v50

Usage: train_w2v.py <data folder with UserReviews.out and WordDict.out> <output file>
"""
import os
import sys

from gensim.models import Word2Vec

folder, out = sys.argv[1:3]
vocab = {l.split("\t")[0] for l in open(os.path.join(folder, "WordDict.out"), encoding="utf-8") if l.strip()}
docs = []
with open(os.path.join(folder, "UserReviews.out"), encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2 and parts[1]:
            docs.append(parts[1].split(" "))
model = Word2Vec(sentences=docs, vector_size=50, sg=1, window=5, min_count=1, workers=1, seed=1, epochs=5)
n = 0
with open(out, "w", encoding="utf-8") as f:
    for w in sorted(vocab):
        if w in model.wv:
            f.write("%s\t%s\n" % (w, " ".join("%.6f" % x for x in model.wv[w])))
            n += 1
print("documents %d, vocabulary %d, vectors written %d" % (len(docs), len(vocab), n))
for probe in ("coffee", "tea", "salty", "expensive"):
    if probe in model.wv:
        print(probe, "->", ", ".join(w for w, _ in model.wv.most_similar(probe, topn=5)))
