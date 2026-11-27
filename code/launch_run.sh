#!/bin/bash
# Start one detached training run.   launch_run.sh <run name> <data variant: full|valsel> <epochs> <seed> [script, default RPMIA_run.py]
W=/root/rpmia
name=$1; variant=$2; epochs=$3; seed=$4; script=${5:-RPMIA_run.py}
mkdir -p $W/runs/$name
# driver library (libcuda.so.1) lives outside the env; prefer the system copy
d=$(ls -d /usr/lib64-nvidia 2>/dev/null || dirname "$(find /usr /opt -xdev -name 'libcuda.so.1' 2>/dev/null | head -1)")
export LD_LIBRARY_PATH=$W/venv/lib:$d
cd $W/src
RPMIA_SEED=$seed RPMIA_EPOCHS=$epochs RPMIA_DATA=$W/data/$variant/ RPMIA_OUT=$W/runs/$name PYTHONPATH=. \
  setsid nohup $W/venv/bin/python src/$script > $W/runs/$name/stdout.log 2>&1 < /dev/null &
echo "launched $name with $script ($variant, $epochs epochs, seed $seed), driver dir $d"
