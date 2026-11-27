#!/bin/bash
# Eval-only text-usage probe on a finished (or running) run's best checkpoint, on CPU so it never
# competes with a training job for GPU memory.   probe_run.sh <run name> <data variant> [script]
W=/root/rpmia
name=$1; variant=$2; script=${3:-RPMIA_run.py}
rm -rf $W/runs/$name/ckpt_probe && cp -r $W/runs/$name/ckpt $W/runs/$name/ckpt_probe
export LD_LIBRARY_PATH=$W/venv/lib
cd $W/src
CUDA_VISIBLE_DEVICES= RPMIA_PROBE=1 RPMIA_CKPT=$W/runs/$name/ckpt_probe/best RPMIA_DATA=$W/data/$variant/ RPMIA_OUT=$W/runs/$name PYTHONPATH=. \
  $W/venv/bin/python src/$script 2>&1 | grep -E "PROBE_RESULT|Error|error" | tee $W/runs/$name/probe.json
