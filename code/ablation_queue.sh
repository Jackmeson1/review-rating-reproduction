#!/bin/bash
# Run a list of paper-equation runs one after another on this box.
#   ablation_queue.sh "<name>:<ablate or full>:<seed>" ...
# Each run: RPMIA_paper.py, validation-selected, at most 40 epochs, stops after 8 epochs without a
# validation gain. Writes runs/<name>/epochs.jsonl and, at the end, queue.done.
W=/root/rpmia
cd $W/src
d=$(ls -d /usr/lib64-nvidia 2>/dev/null)
export LD_LIBRARY_PATH=$W/venv/lib:$d
rm -f $W/queue.done
for spec in "$@"; do
  IFS=: read -r name ablate seed <<< "$spec"
  [ "$ablate" = full ] && ablate=""
  export RPMIA_EXCLUDE_TARGET=""
  [ "$ablate" = notarget ] && { ablate=""; export RPMIA_EXCLUDE_TARGET=1; }  # full model, target review left out in training
  mkdir -p $W/runs/$name
  echo "$(date +%H:%M:%S) start $name (ablate='$ablate', seed $seed)" >> $W/logs/queue.log
  RPMIA_ABLATE=$ablate RPMIA_SEED=$seed RPMIA_EPOCHS=40 RPMIA_PATIENCE=8 RPMIA_EMB=$W/data/WordEmbedding.out \
    RPMIA_DATA=$W/data/valsel/ RPMIA_OUT=$W/runs/$name PYTHONPATH=. \
    $W/venv/bin/python src/RPMIA_paper.py > $W/runs/$name/stdout.log 2>&1
  echo "$(date +%H:%M:%S) done  $name exit $?" >> $W/logs/queue.log
done
touch $W/queue.done
