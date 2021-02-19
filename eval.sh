#!/usr/bin/env bash

set -x

CONFIG_FILE=$1
CONFIG_PY="${CONFIG_FILE##*/}"
CONFIG="${CONFIG_PY%.*}"
WORK_DIR="./work_dirs/${CONFIG}"
TMPDIR="${WORK_DIR}/tmp"
CHECKPOINT="${WORK_DIR}/latest.pth"
RESULT_FILE="${WORK_DIR}/result.pkl"

# train config
GPUS=4
PORT=${PORT:-29511}
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\nconfig file: ${CONFIG}\n"

# evaluation
echo -e "\nEvaluating ${WORK_DIR}\n"
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
  ./tools/test.py \
  ${CONFIG_FILE} \
  ${CHECKPOINT} \
  --launcher 'pytorch' \
  --work-dir $WORK_DIR \
  --eval mIoU  \
  --tmpdir $TMPDIR \
#  --out $RESULT_FILE \

echo -e "\nWork Dir: ${WORK_DIR}.\n"