#!/bin/bash
# Exact stack from the repo README, on Python 3.7 (the newest Python with TF 1.15 wheels).
#   bash setup_env.sh cpu|gpu
# gpu adds CUDA 10.0 + cuDNN 7.6 inside the env (what the TF 1.15.0 wheel was built against).
# scipy is imported by _auxiliaryTools/ExtractData.py but missing from the README.
set -uo pipefail
mode=${1:-cpu}
W=/root/rpmia
mkdir -p $W/logs $W/raw $W/data $W/runs
exec > $W/logs/setup.log 2>&1
rm -f $W/setup.done $W/setup.failed
fail() { echo "FAILED: $1"; touch $W/setup.failed; exit 1; }
[ -d $W/src/.git ] || git clone -q https://github.com/INFORMSJoC/2023.0131.git $W/src
git -C $W/src checkout -q 564c27ca9510daafccf5a4bf4fa5170a50ec6a24 && git -C $W/src log -1 --format='src %H %s'
if [ ! -x $W/bin/micromamba ]; then
  mkdir -p $W/bin && curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $W bin/micromamba || fail micromamba
fi
export MAMBA_ROOT_PREFIX=$W/mamba
[ -x $W/venv/bin/python ] || $W/bin/micromamba create -y -q -p $W/venv -c conda-forge python=3.7 pip || fail python
tf=tensorflow
if [ "$mode" = gpu ]; then
  tf=tensorflow-gpu
  $W/bin/micromamba install -y -q -p $W/venv -c conda-forge cudatoolkit=10.0 cudnn=7.6 || fail cuda
fi
$W/venv/bin/python -m pip install -q --progress-bar off $tf==1.15.0 numpy==1.19.2 pandas==1.1.5 scipy==1.5.4 'protobuf<=3.20.3' || fail pip
export LD_LIBRARY_PATH=$W/venv/lib
$W/venv/bin/python -c "import tensorflow as tf, numpy, scipy, pandas; print('tf', tf.__version__, 'np', numpy.__version__, 'scipy', scipy.__version__, 'pd', pandas.__version__); print('gpu', tf.test.is_gpu_available())" 2>&1 | grep -E "^(tf|gpu) |Could not load|Created TensorFlow device" || fail import
$W/venv/bin/python -m pip freeze > $W/logs/pip-freeze.txt
# data: raw Amazon 2014 5-core Grocery file, then the two review files the repo omits
[ -s $W/raw/g5.json.gz ] || curl -sS --max-time 300 -o $W/raw/g5.json.gz "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz" || fail download
echo "77c5b65fb395dcd24f6a81d95d2ea471e8d9a82b70431b39919c14ab52547dd1  $W/raw/g5.json.gz" | sha256sum -c - || fail checksum
python3 $W/build_reviews.py $W/src/dataset/dataset_G $W/raw/g5.json.gz $W/data > $W/logs/build_reviews.log 2>&1 || fail build
tail -1 $W/logs/build_reviews.log
python3 $W/make_run.py $W/src/src/RPMIA.py $W/src/src/RPMIA_run.py || fail patch
echo ok > $W/setup.done
