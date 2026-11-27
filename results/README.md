# Results

`runs/<name>/epochs.jsonl` has one JSON line per training epoch: training error, validation MSE and
MAE, test MSE and MAE. `runs/<name>/probe.json`, where present, is the shuffle test on that run's best
checkpoint. `summarize.py` reads all of it and writes `summary.json`. Every figure is generated from that file, and the
README tables are copied from its printed output.

All runs train on the 70% split and select the epoch on the 10% validation split, except where noted.

| Run folder | README row | Script |
|---|---|---|
| `gpu_valsel_s1` | Released code, as shipped (30 epochs) | `make_run.py` output, no flags |
| `gpu_valsel_l2fix_s1` | Released code, bracket repaired only | `make_run.py --l2-inside` |
| `l2fix_fmfix_200` | Released code, both repairs (84 epochs; first 30 used for the matched row) | `make_run.py --l2-inside --fm-fix` |
| `eq_s1`, `full_s2`, `full_s3` | Paper's equations, seeds 1 to 3 | `RPMIA_paper.py` |
| `eq_bias_s1` | Paper's equations + user and item bias | `RPMIA_paper.py`, `RPMIA_BIAS=1` |
| `context_s1..3` | Context module removed | `RPMIA_paper.py`, `RPMIA_ABLATE=context` |
| `aspect_s1..3` | Aspect module removed | `RPMIA_paper.py`, `RPMIA_ABLATE=aspect` |
| `uiap_s1..3` | Aspect-pair module removed | `RPMIA_paper.py`, `RPMIA_ABLATE=uiap` |
| `notarget_s1..3` | Own review hidden in training | `RPMIA_paper.py`, `RPMIA_EXCLUDE_TARGET=1` |
| `cpu_valsel` | not in the README; two CPU epochs used to time the code | `make_run.py` output, no flags |

Notes

- `eq_s1` ran for 39 epochs without early stopping. The other paper-equation runs stop after 8 epochs
  without a validation gain, with a cap of 40. All are scored at their best validation epoch, so the
  difference in length does not change the score.
- The released-code runs log `train_loss`, which includes the regularisation terms. The
  paper-equation runs log `train_mse`. The two are not comparable with each other.
- The shuffle test for "as shipped" was run on an epoch-12 checkpoint of a separate run of the same
  script, so its test MSE (1.130) differs from the 30-epoch row (1.092).
- `floor.json` holds the no-text baselines from `code/floor.py`: the global mean, user and item
  biases with damping chosen on validation, and biased matrix factorisation.
- `logs/` holds the input rebuild, the vocabulary check, the Word2Vec training and the baseline logs.
