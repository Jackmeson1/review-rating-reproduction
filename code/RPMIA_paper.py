# coding: utf-8
"""RPMIA as the paper's equations describe it (Yu et al. 2025, INFORMS JoC, section 3), on TensorFlow 1.15.

The released src/RPMIA.py implements a different graph (see the table below). This file follows the
paper. Where the paper is explicit it wins; where it is silent the released code's choice is kept; where
both are silent, or the paper's form cannot be built, the choice is marked FILL. All fills were fixed
before any test result of this file was seen.

  Paper                                     | Released code                     | Here
  ------------------------------------------+-----------------------------------+---------------------------------
  Eq.1  pretrained Word2Vec, 50-d           | random uniform [0,1), 300-d       | paper; source not given, FILL:
                                            |                                   | Word2Vec trained on the training
                                            |                                   | documents (train_w2v.py)
  Eq.2  conv window 3, 50 filters, bias     | same, no bias                     | paper
  Eq.3-7 H-head cross attention against the | one head, bilinear form against   | paper; H not given, FILL H=2;
        other side's mean context, softmax  | every position, padding included  | FILL: padding masked out of the
        over terms, heads concatenated      |                                   | mean and the softmax
  Eq.8-9 gate sigmoid(R_p * W C + b),       | gate sigmoid(C W + A_p W' + b)    | paper; W_i read as one matrix per
        R_p = sigmoid(H_p A_p + b_p)        |                                   | aspect (a matrix per term i is
                                            |                                   | not buildable), FILL
  Eq.10 mean, dot-product softmax, sum      | capsule routing with squash       | paper
  Eq.11 (add) || (dot) || (cross product)   | (multiply) || (subtract)          | add || elementwise product ||
                                            |                                   | subtract. FILL: an outer product
                                            |                                   | is D*D per pair and cannot feed
                                            |                                   | Eq.15; the released subtract
                                            |                                   | stands in for it
  Eq.12-13 softmax against the mean pair,   | additive attention, pairs summed  | paper
        25 weighted pairs concatenated      |                                   |
  Eq.15 FM on Z with a full matrix M        | FM on a 50-d projection of Z,     | FM on Z; M = V V^T of rank 50,
                                            | factorised                        | FILL (full M is 7500 x 7500)
  Eq.16 FNN, depth and width not given      | one ReLU layer of 50              | released: one ReLU layer of 50
  Eq.14 sigmoid(W_wide FM + W_deep FNN + b) | FM + user bias + item bias, no    | paper; FILL: 1 + 4*sigmoid(.) so
        no user or item bias                | sigmoid                           | the output spans the 1-5 scale
  Eq.17 summed squared error + 1e-6 |Θ|^2,  | mean squared error, L2 on a few   | paper
        RMSprop 1e-3, batch 256             | weights (weight 1 by a misplaced  |
                                            | bracket), batch 200               |
  dropout rate 0.2, placement not given     | keep 0.5 applied to weight        | FILL: rate 0.2 on Z and on the
                                            | matrices                          | FNN hidden layer
  minibatch order not given                 | fixed file order (sorted by user) | FILL: reshuffled every epoch
  epochs / stopping not given               | 200 epochs, test MSE each epoch   | FILL: epoch chosen on validation

Environment variables: RPMIA_DATA (folder), RPMIA_OUT, RPMIA_EPOCHS, RPMIA_SEED, RPMIA_EMB (Word2Vec
file, empty = random normal), RPMIA_HEADS, RPMIA_ABLATE (context|aspect|uiap, the paper's Table 7),
RPMIA_BIAS=1 (add user and item bias, not in the paper), RPMIA_PROBE=1 (eval-only shuffle test),
RPMIA_MAX_STEPS (timing probe).
"""
import json
import math
import os
import random
import sys
from time import time

import numpy as np
import tensorflow as tf

from _auxiliaryTools.ExtractData import Dataset
from _auxiliaryTools.Evaluation import get_test_list_mask

env = os.environ.get
DATA = os.environ["RPMIA_DATA"]
OUT = env("RPMIA_OUT", ".")
EPOCHS = int(env("RPMIA_EPOCHS", 200))
SEED = int(env("RPMIA_SEED", 1))
HEADS = int(env("RPMIA_HEADS", 2))
ABLATE = env("RPMIA_ABLATE", "")
USE_BIAS = env("RPMIA_BIAS", "") == "1"
EMB_PATH = env("RPMIA_EMB", "")
MAX_STEPS = int(env("RPMIA_MAX_STEPS", 0))
PATIENCE = int(env("RPMIA_PATIENCE", 0))  # 0 = run all epochs
assert ABLATE in ("", "context", "aspect", "uiap")

# section 3.8
BATCH = 256
LR = 1e-3
L2 = 1e-6
DROP_RATE = 0.2
WORD_DIM = 50
LATENT = 50          # FM rank and FNN width
FILTERS = 50
WINDOW = 3
MAX_LEN = 300
K = 5                # aspects per user and per item (section 3.5)
NEG = -1e9

np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)
os.makedirs(OUT, exist_ok=True)

t0 = time()
data = Dataset(MAX_LEN, DATA)
word_dict, user_reviews, item_reviews = data.word_id_dict, data.userReview_dict, data.itemReview_dict
user_masks, item_masks, train, test_ratings = data.userMask_dict, data.itemMask_dict, data.trainMtrx, data.testRatings
num_users, num_items = train.shape
vocab = len(word_dict)
print("load data: %.1fs, users %d, items %d, vocabulary %d" % (time() - t0, num_users, num_items, vocab), flush=True)

if EMB_PATH:
    emb_init = np.random.normal(0.0, 0.1, (vocab, WORD_DIM))
    found = 0
    with open(EMB_PATH, encoding="utf-8") as f:
        for line in f:
            w, vec = line.rstrip("\n").split("\t")
            if w in word_dict:
                emb_init[word_dict[w]] = np.array(vec.split(" "), dtype=np.float32)
                found += 1
    print("word vectors loaded for %d of %d words" % (found, vocab), flush=True)
else:
    emb_init = np.random.normal(0.0, 0.1, (vocab, WORD_DIM))

# ---- graph --------------------------------------------------------------------------------------
users = tf.placeholder(tf.int32, [None], name="users")
items = tf.placeholder(tf.int32, [None], name="items")
u_ids = tf.placeholder(tf.int32, [None, MAX_LEN], name="u_ids")
i_ids = tf.placeholder(tf.int32, [None, MAX_LEN], name="i_ids")
u_mask = tf.placeholder(tf.float32, [None, MAX_LEN], name="u_mask")
i_mask = tf.placeholder(tf.float32, [None, MAX_LEN], name="i_mask")
ratings = tf.placeholder(tf.float32, [None, 1], name="ratings")
keep = tf.placeholder(tf.float32, name="keep")


def tn(shape, std, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=std), name=name)


def masked_mean(x, mask):  # x [B,L,D], mask [B,L] -> [B,D]
    m = tf.expand_dims(mask, -1)
    return tf.reduce_sum(x * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)


def masked_softmax(score, mask):  # over axis 1; mask [B,L] broadcast over trailing dims of score
    while mask.shape.ndims < score.shape.ndims:
        mask = tf.expand_dims(mask, -1)
    return tf.nn.softmax(score + (1.0 - mask) * NEG, axis=1)


# Eq.1 embedding; the padding id (== vocab) maps to a fixed zero row
embedding = tf.concat([tf.Variable(emb_init, dtype=tf.float32, name="word_embedding"), tf.zeros([1, WORD_DIM])], 0)


def preliminary_context(ids, mask, side):  # Eq.2
    x = tf.nn.embedding_lookup(embedding, ids)
    w = tn([WINDOW, WORD_DIM, FILTERS], 0.3, "conv_W_" + side)
    b = tf.Variable(tf.zeros([FILTERS]), name="conv_b_" + side)
    return tf.nn.relu(tf.nn.conv1d(x, w, stride=1, padding="SAME") + b) * tf.expand_dims(mask, -1)


def cross_attention(c, mask, other_mean, side):  # Eq.3-7
    joined = tf.concat([c, tf.tile(tf.expand_dims(other_mean, 1), [1, MAX_LEN, 1])], -1)      # [B,L,2F]
    w = tf.Variable(tf.glorot_uniform_initializer()([2 * FILTERS, HEADS]), name="xatt_W_" + side)
    b = tf.Variable(tf.zeros([HEADS]), name="xatt_b_" + side)
    lam = tf.tanh(tf.tensordot(joined, w, axes=1) + b)                                         # [B,L,H]
    alpha = masked_softmax(lam, mask)                                                          # over terms
    return tf.concat([c * alpha[:, :, h:h + 1] for h in range(HEADS)], -1), alpha              # [B,L,H*F]


def aspects(c, mask, side):  # Eq.8-10 -> list of K tensors [B,D]
    d = int(c.shape[-1])
    out = []
    for p in range(K):
        a_p = tn([1, d], 0.3, "%s_aspect_%d" % (side, p))
        h_p = tn([d, d], 0.1, "%s_H_%d" % (side, p))
        b_p = tf.Variable(tf.zeros([d]), name="%s_bH_%d" % (side, p))
        r_p = tf.sigmoid(tf.matmul(a_p, h_p) + b_p)                                            # Eq.8, [1,d]
        w_p = tn([d, d], 0.1, "%s_W_%d" % (side, p))
        b2 = tf.Variable(tf.zeros([d]), name="%s_bW_%d" % (side, p))
        cand = c * tf.sigmoid(r_p * tf.tensordot(c, w_p, axes=1) + b2) * tf.expand_dims(mask, -1)  # Eq.9
        mean = masked_mean(cand, mask)                                                         # Eq.10
        beta = masked_softmax(tf.reduce_sum(cand * tf.expand_dims(mean, 1), -1), mask)
        out.append(tf.reduce_sum(cand * tf.expand_dims(beta, -1), 1))
    return out


cu0 = preliminary_context(u_ids, u_mask, "u")
ci0 = preliminary_context(i_ids, i_mask, "i")
if ABLATE == "context":      # Table 7, RPMIA-Context: preliminary contexts go straight on
    cu, ci, alpha_u, alpha_i = cu0, ci0, None, None
else:
    cu, alpha_u = cross_attention(cu0, u_mask, masked_mean(ci0, i_mask), "u")
    ci, alpha_i = cross_attention(ci0, i_mask, masked_mean(cu0, u_mask), "i")

if ABLATE == "aspect":       # Table 7, RPMIA-Aspect: one vector per side, the summed attentive contexts
    au = [tf.reduce_sum(cu * tf.expand_dims(u_mask, -1), 1)]
    ai = [tf.reduce_sum(ci * tf.expand_dims(i_mask, -1), 1)]
else:
    au = aspects(cu, u_mask, "u")
    ai = aspects(ci, i_mask, "i")

if ABLATE == "uiap":         # Table 7, RPMIA-UIAP: aspects of both sides concatenated, no pairing
    z = tf.concat(au + ai, -1)
    gamma = None
else:                        # Eq.11-13
    pairs = tf.stack([tf.concat([a + b, a * b, a - b], -1) for a in au for b in ai], 1)        # [B,K*K,3D]
    gamma = tf.nn.softmax(tf.reduce_sum(pairs * tf.reduce_mean(pairs, 1, keepdims=True), -1), axis=1)
    z = tf.reshape(pairs * tf.expand_dims(gamma, -1), [-1, int(pairs.shape[1]) * int(pairs.shape[2])])

n_z = int(z.shape[-1])
x = tf.nn.dropout(z, keep)
# Eq.15 factorisation machine, M = V V^T
m0 = tf.Variable(tf.zeros([1]), name="fm_m0")
m = tn([n_z, 1], 0.3, "fm_m")
v = tn([n_z, LATENT], 0.3, "fm_V")
phi = m0 + tf.matmul(x, m) + 0.5 * tf.reduce_sum(tf.square(tf.matmul(x, v)), 1, keepdims=True)
# Eq.16 feedforward network
w1 = tf.Variable(tf.random_normal([n_z, LATENT], stddev=0.02), name="fnn_W1")
b1 = tf.Variable(tf.constant(0.1, shape=[LATENT]), name="fnn_b1")
psi = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w1) + b1), keep)
# Eq.14
w_wide = tf.Variable(tf.ones([1]), name="w_wide")
w_deep = tn([LATENT, 1], 0.3, "w_deep")
b_r = tf.Variable(tf.zeros([1]), name="b_r")
logit = w_wide * phi + tf.matmul(psi, w_deep) + b_r
if USE_BIAS:                 # not in the paper
    user_bias = tf.Variable(tf.zeros([num_users, 1]), name="user_bias")
    item_bias = tf.Variable(tf.zeros([num_items, 1]), name="item_bias")
    logit += tf.nn.embedding_lookup(user_bias, users) + tf.nn.embedding_lookup(item_bias, items)
predict_rating = 1.0 + 4.0 * tf.sigmoid(logit)

# Eq.17
sq_err = tf.square(predict_rating - ratings)
l2_all = tf.add_n([tf.reduce_sum(tf.square(p)) for p in tf.trainable_variables()])
sq_sum = tf.reduce_sum(sq_err)
loss = sq_sum + L2 * l2_all
train_step = tf.train.RMSPropOptimizer(LR).minimize(loss)
n_params = int(sum(np.prod(p.shape.as_list()) for p in tf.trainable_variables()))
print("ablate=%r heads=%d z=%d params=%d bias=%s emb=%s" % (ABLATE, HEADS, n_z, n_params, USE_BIAS, EMB_PATH or "random"), flush=True)

# ---- data feeding ---------------------------------------------------------------------------------
train_rows = [(u, i, train[u, i]) for (u, i) in train.keys()]


EXCLUDE_TARGET = env("RPMIA_EXCLUDE_TARGET", "") == "1"
if EXCLUDE_TARGET:
    # Diagnostic, not in the paper: while TRAINING on a pair (u, i), rebuild both documents without the
    # review u wrote on i, then take the first 300 words as usual. Validation and test documents are the
    # standard ones (their target review was never in them). Needs ReviewTokens.out from build_reviews.py.
    by_user, by_item = {}, {}
    with open(os.path.join(DATA, "ReviewTokens.out"), encoding="utf-8") as f:
        for line in f:
            u, i, toks = line.rstrip("\n").split("\t")
            ids = [word_dict[w] for w in toks.split(" ") if w in word_dict]
            by_user.setdefault(int(u), []).append((int(i), ids))
            by_item.setdefault(int(i), []).append((int(u), ids))

    def doc_without(reviews, other):
        ids = [t for o, r in reviews if o != other for t in r][:MAX_LEN]
        return ids + [vocab] * (MAX_LEN - len(ids)), [1.0] * len(ids) + [0.0] * (MAX_LEN - len(ids))

    # the review file must reproduce the standard documents exactly when nothing is left out
    for u in list(by_user)[:200]:
        assert doc_without(by_user[u], -1)[0] == user_reviews[u], "ReviewTokens.out does not match UserReviews.out"
    for i in list(by_item)[:200]:
        assert doc_without(by_item[i], -1)[0] == item_reviews[i], "ReviewTokens.out does not match ItemReviews.out"
    loo = {}
    changed_u = changed_i = 0
    for (u, i, _) in train_rows:
        du, mu_ = doc_without(by_user[u], i)
        di, mi_ = doc_without(by_item[i], u)
        changed_u += du != user_reviews[u]
        changed_i += di != item_reviews[i]
        loo[(u, i)] = (du, mu_, di, mi_)
    print("target excluded in training: user document changed for %d of %d pairs, item document for %d"
          % (changed_u, len(train_rows), changed_i), flush=True)


def feed_train(rows, keep_prob):
    if not EXCLUDE_TARGET:
        return feed(rows, user_reviews, user_masks, item_reviews, item_masks, keep_prob)
    docs = [loo[(r[0], r[1])] for r in rows]
    return {users: [r[0] for r in rows], items: [r[1] for r in rows],
            u_ids: [d[0] for d in docs], u_mask: [d[1] for d in docs],
            i_ids: [d[2] for d in docs], i_mask: [d[3] for d in docs],
            ratings: [[r[2]] for r in rows], keep: keep_prob}


def feed(rows, ur, um, ir, im, keep_prob):
    return {users: [r[0] for r in rows], items: [r[1] for r in rows],
            u_ids: [ur[r[0]] for r in rows], u_mask: [um[r[0]] for r in rows],
            i_ids: [ir[r[1]] for r in rows], i_mask: [im[r[1]] for r in rows],
            ratings: [[r[2]] for r in rows], keep: keep_prob}


def evaluate(sess, rows, ur=user_reviews, um=user_masks, ir=item_reviews, im=item_masks):
    se = ae = 0.0
    for s in range(0, len(rows), 512):
        part = rows[s:s + 512]
        p = sess.run(predict_rating, feed(part, ur, um, ir, im, 1.0))[:, 0]
        r = np.array([x[2] for x in part])
        se += float(np.sum((p - r) ** 2))
        ae += float(np.sum(np.abs(p - r)))
    return se / len(rows), ae / len(rows)


val_path = os.path.join(DATA, "ValidateInteraction.out")
val_ratings = data.load_ratingFile_as_list(val_path) if os.path.exists(val_path) else None
saver = tf.train.Saver(max_to_keep=1)
steps = int(math.ceil(len(train_rows) / float(BATCH)))
print("train rows %d, steps/epoch %d, epochs %d, validation %s" % (len(train_rows), steps, EPOCHS, val_ratings is not None), flush=True)

if env("RPMIA_PROBE"):
    def shuffled(docs, masks, seed):
        keys = sorted(docs)
        src = keys[:]
        random.Random(seed).shuffle(src)
        return {k: docs[s] for k, s in zip(keys, src)}, {k: masks[s] for k, s in zip(keys, src)}

    with tf.Session() as sess:
        saver.restore(sess, env("RPMIA_CKPT", os.path.join(OUT, "ckpt", "best")))
        res = {"test_mse_real_docs": evaluate(sess, test_ratings)[0]}
        ur, um = shuffled(user_reviews, user_masks, 1)
        ir, im = shuffled(item_reviews, item_masks, 2)
        res["test_mse_shuffled_docs"] = evaluate(sess, test_ratings, ur, um, ir, im)[0]
        res["test_mse_shuffled_user_docs_only"] = evaluate(sess, test_ratings, ur, um, item_reviews, item_masks)[0]
        res["test_mse_shuffled_item_docs_only"] = evaluate(sess, test_ratings, user_reviews, user_masks, ir, im)[0]
        print("PROBE_RESULT " + json.dumps(res), flush=True)
    sys.exit(0)

best = None
rng = random.Random(SEED)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(EPOCHS):
        t = time()
        rng.shuffle(train_rows)
        loss_total = 0.0
        for s in range(steps):
            part = train_rows[s * BATCH:(s + 1) * BATCH]
            _, loss_val = sess.run([train_step, sq_sum], feed_train(part, 1.0 - DROP_RATE))
            loss_total += loss_val
            if MAX_STEPS and s + 1 == MAX_STEPS:
                dt = time() - t
                print("PROBE %d steps in %.1fs -> %.3fs/step, est %.1f min/epoch, train mse so far %.4f"
                      % (MAX_STEPS, dt, dt / MAX_STEPS, dt / MAX_STEPS * steps / 60, loss_total / (MAX_STEPS * BATCH)), flush=True)
                sys.exit(0)
        t1 = time()
        rec = {"epoch": e, "train_mse": loss_total / len(train_rows), "train_s": round(t1 - t, 1)}
        rec["test_mse"], rec["test_mae"] = evaluate(sess, test_ratings)
        if val_ratings is not None:
            rec["val_mse"], rec["val_mae"] = evaluate(sess, val_ratings)
        rec["eval_s"] = round(time() - t1, 1)
        key = rec.get("val_mse", rec["test_mse"])
        if not math.isfinite(rec["train_mse"]):
            rec["diverged"] = True
        if best is None or key < best:
            best = key
            best_epoch = e
            rec["saved"] = True
            saver.save(sess, os.path.join(OUT, "ckpt", "best"), write_meta_graph=False)
        with open(os.path.join(OUT, "epochs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if rec.get("diverged"):
            sys.exit(3)
        if PATIENCE and e - best_epoch >= PATIENCE:  # no validation gain for PATIENCE epochs
            print("early stop at epoch %d, best validation at epoch %d" % (e, best_epoch), flush=True)
            break
