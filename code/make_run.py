#!/usr/bin/env python3
"""Write RPMIA_run.py: the repo's src/RPMIA.py with the model graph untouched and only the harness changed.

Changes, each asserted to apply exactly once:
  1. data folder and epoch count come from the environment (RPMIA_DATA, RPMIA_EPOCHS)
  2. optional seed (RPMIA_SEED); the original sets none
  3. the training loop is replaced by one that does the same optimiser step per batch, but
     - logs one JSON line per epoch (train loss, test MSE/MAE, and validation MSE when the data
       folder has ValidateInteraction.out)
     - drops the per-epoch Saver (a new Saver object and a ~24 MB checkpoint every epoch) and the
       var_epoch_*.txt dumps; it keeps one checkpoint, the best epoch so far
Everything from the top of the file to the optimiser line is the authors' code, byte for byte.

Optional 4th change, only with --l2-inside: the shipped loss line closes the lambda_1 bracket after the
bias terms, so the L2 penalty on the conv, gate and projection weights enters the loss with weight 1
instead of lambda_1 = 0.001. The flag moves that bracket to the end of the line. Nothing else changes.

Optional --paper-settings (use together with --l2-inside): the hyperparameters stated in the paper's
section 3.8 instead of the ones in the released script.

Usage: make_run.py <path to src/RPMIA.py> <output path> [--l2-inside] [--paper-settings]
"""
import sys

src_path, out_path = sys.argv[1:3]
code = open(src_path, encoding="utf-8").read()


def swap(old, new):
    global code
    assert code.count(old) == 1, "expected exactly one occurrence of: %r" % old
    code = code.replace(old, new)


swap("    epochs = 200\n", "    epochs = int(os.environ.get('RPMIA_EPOCHS', 200))\n")
swap('    dataSet = Dataset(max_doc_length, "./outputs/output_G/") #!!!!!在这里更改数据所在目录!!!!!\n',
     "    dataSet = Dataset(max_doc_length, os.environ['RPMIA_DATA'])\n")
swap("epsilon = 1e-9\n",
     "epsilon = 1e-9\n"
     "import os, json\n"
     "if os.environ.get('RPMIA_SEED'):\n"
     "    np.random.seed(int(os.environ['RPMIA_SEED'])); tf.set_random_seed(int(os.environ['RPMIA_SEED']))\n")

if "--l2-inside" in sys.argv:
    swap("tf.nn.l2_loss(item_bs))+ tf.nn.l2_loss(W_conv_u)", "tf.nn.l2_loss(item_bs)+ tf.nn.l2_loss(W_conv_u)")
    swap("+tf.nn.l2_loss(W_prj_u)+tf.nn.l2_loss(W_prj_i)\n", "+tf.nn.l2_loss(W_prj_u)+tf.nn.l2_loss(W_prj_i))\n")

if "--paper-settings" in sys.argv:
    # Section 3.8 of the paper, applied to the released graph: batch 256, L2 coefficient 1e-6,
    # dropout rate 0.2 (the code's drop_out is a keep probability, so 0.8), word vectors of 50 dimensions.
    swap("    batch_size = 200\n", "    batch_size = 256\n")
    swap("    lambda_1 = 0.001\n", "    lambda_1 = 1e-6\n")
    swap("    drop_out = 0.5\n", "    drop_out = 0.8\n")
    swap("    word_latent_dim = 300#denote as d in paper\n", "    word_latent_dim = 50\n")

if "--fm-fix" in sys.argv:
    # J_2 is [B,1] but tf.trace returns [B], so J_2 - J_3 broadcasts to [B,B]: every rating in a batch is
    # compared with a whole row of predictions, and evaluation reads column 0 only. Give the trace its axis.
    swap("    J_total = (J_1 + 0.5 * (J_2 - J_3))\n", "    J_total = J_1 + 0.5 * (J_2 - J_3[:, None])\n")

marker = "train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)\n"
assert code.count(marker) == 1
head = code[: code.index(marker) + len(marker)]

tail = r'''
# ---- harness below replaces the original loop; the optimiser step is the same ----
out_dir = os.environ.get('RPMIA_OUT', '.')
os.makedirs(out_dir, exist_ok=True)
val_path = os.path.join(os.environ['RPMIA_DATA'], 'ValidateInteraction.out')
val = None
if os.path.exists(val_path):
    val = get_test_list_mask(200, dataSet.load_ratingFile_as_list(val_path), user_reviews, item_reviews, user_masks, item_masks)
max_steps = int(os.environ.get('RPMIA_MAX_STEPS', 0))  # >0: timing probe, stop after that many steps


def run_eval(sess, batches):
    u_b, i_b, ui_b, ii_b, um_b, im_b, r_b = batches
    sq, ab = [], []
    for k in range(len(ui_b)):
        eval_model(users, items, users_inputs, items_inputs, users_masks_inputs, items_masks_inputs,
                   dropout_rate, b_inputs, predict_rating, sess, u_b[k], i_b[k], ui_b[k], ii_b[k],
                   um_b[k], im_b[k], r_b[k], np.zeros([len(ui_b[k]), 1, 1]), sq, ab)
    return float(np.mean(sq)), float(np.mean(ab))


test = (user_test, item_test, user_input_test, item_input_test, user_mask_test, item_mask_test, rating_input_test)
saver = tf.train.Saver(max_to_keep=1)
steps_per_epoch = int(math.ceil(len(user_input) / float(batch_size)))
print("train rows %d, steps/epoch %d, epochs %d, validation %s" % (len(user_input), steps_per_epoch, epochs, val is not None), flush=True)
if os.environ.get('RPMIA_PROBE'):
    # Eval-only: does the trained model use the review text at all? Score the test set with the real
    # documents, then with documents shuffled across users and across items. Equal MSEs = text path dead.
    import random

    def shuffled(docs, masks, seed):
        keys = sorted(docs)
        src = keys[:]
        random.Random(seed).shuffle(src)
        return {k: docs[v] for k, v in zip(keys, src)}, {k: masks[v] for k, v in zip(keys, src)}

    text_l2 = tf.nn.l2_loss(W_conv_u) + tf.nn.l2_loss(W_conv_i) + tf.nn.l2_loss(W_word_gate_u) + tf.nn.l2_loss(W_word_gate_i) + tf.nn.l2_loss(W_prj_u) + tf.nn.l2_loss(W_prj_i)
    with tf.Session() as sess:
        saver.restore(sess, os.environ.get('RPMIA_CKPT', os.path.join(out_dir, "ckpt", "best")))
        res = {}
        res["test_mse_real_docs"], _ = run_eval(sess, test)
        ur, um = shuffled(user_reviews, user_masks, 1)
        ir, im = shuffled(item_reviews, item_masks, 2)
        res["test_mse_shuffled_docs"], _ = run_eval(sess, get_test_list_mask(200, testRatings, ur, ir, um, im))
        l2, ub, ib, w0 = sess.run([text_l2, user_bias, item_bias, w_0])
        res["unscaled_l2_of_text_weights"] = float(l2)
        res["user_bias_std"], res["item_bias_std"], res["w_0"] = float(ub.std()), float(ib.std()), float(w0[0])
        jt = sess.run(J_total, feed_dict={users: user_test[0], items: item_test[0], users_inputs: user_input_test[0],
                                          items_inputs: item_input_test[0], users_masks_inputs: user_mask_test[0],
                                          items_masks_inputs: item_mask_test[0], b_inputs: np.zeros([len(user_input_test[0]), 1, 1]),
                                          dropout_rate: 1.0})
        res["text_score_mean_first_batch"], res["text_score_std_first_batch"] = float(jt.mean()), float(jt.std())
        print("PROBE_RESULT " + json.dumps(res), flush=True)
    sys.exit(0)

best = None
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        t = time()
        loss_total = 0.0
        for i in range(steps_per_epoch):
            user_batch, item_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, rates_batch = get_train_instance_batch(
                i, batch_size, user_input, item_input, rateings, user_reviews, item_reviews, user_masks, item_masks)
            _, loss_val = sess.run([train_step, loss],
                                   feed_dict={users: user_batch, items: item_batch,
                                              users_inputs: user_input_batch, items_inputs: item_input_batch,
                                              users_masks_inputs: user_mask_batch, items_masks_inputs: item_mask_batch,
                                              ratings: rates_batch, b_inputs: b_train, dropout_rate: drop_out})
            loss_total += loss_val
            if max_steps and i + 1 == max_steps:
                print("PROBE %d steps in %.1fs -> %.3fs/step, est %.1f min/epoch" % (max_steps, time() - t, (time() - t) / max_steps, (time() - t) / max_steps * steps_per_epoch / 60), flush=True)
                sys.exit(0)
        t1 = time()
        rec = {"epoch": e, "train_loss": loss_total / steps_per_epoch, "train_s": round(t1 - t, 1)}
        rec["test_mse"], rec["test_mae"] = run_eval(sess, test)
        if val is not None:
            rec["val_mse"], rec["val_mae"] = run_eval(sess, val)
        rec["eval_s"] = round(time() - t1, 1)
        key = rec.get("val_mse", rec["test_mse"])
        if best is None or key < best:
            best = key
            rec["saved"] = True
            saver.save(sess, os.path.join(out_dir, "ckpt", "best"), write_meta_graph=False)
        with open(os.path.join(out_dir, "epochs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
'''
head = head.replace("import math\n", "import math\nimport sys\n", 1)
open(out_path, "w", encoding="utf-8").write(head + tail)
print("wrote", out_path)
