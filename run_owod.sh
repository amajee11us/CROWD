#!/bin/bash
set -euo pipefail  # Exit immediately if any command fails, if any variable is undefined, or if a pipeline fails

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}
GPUS=${GPUS:-"4,5,6,7"}
BASELINE=${BASELINE:-"baseline_FL_FLCG_mod"}

if [ $BENCHMARK == "M-OWODB" ]; then
  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                                --num-gpus 4 \
                                --dist-url tcp://127.0.0.1:${PORT} \
                                --task ${BENCHMARK}/t1 \
                                --config-file configs/${BENCHMARK}/t1.yaml \
                                OUTPUT_DIR output/${BASELINE}/${BENCHMARK}/t1/

  CUDA_VISIBLE_DEVICES=${GPUS} python discover_unknown.py \
                                --config-file configs/${BENCHMARK}/t1_ft.yaml \
                                --input-txt datasets/ImageSets/Main/${BENCHMARK}/t1.txt \
                                --task ${BENCHMARK}/t1 \
                                --output output/${BASELINE}/${BENCHMARK}/t1/unknown_rois.json \
                                --opts MODEL.WEIGHTS output/${BASELINE}/${BENCHMARK}/t1/model_0009999.pth DISCOVER_UNKNOWN True 

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                                --num-gpus 4 \
                                --dist-url tcp://127.0.0.1:${PORT} \
                                --task ${BENCHMARK}/t1 \
                                --config-file configs/${BENCHMARK}/t1_ft.yaml \
                                --resume \
                                MODEL.WEIGHTS output/${BASELINE}/${BENCHMARK}/t1/model_0009999.pth \
                                DISCOVER_STORE_PATH output/${BASELINE}/${BENCHMARK}/t1/unknown_rois.json \
                                OUTPUT_DIR output/${BASELINE}/${BENCHMARK}/t1/

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                                --num-gpus 4 \
                                --dist-url tcp://127.0.0.1:${PORT} \
                                --task ${BENCHMARK}/t2 \
                                --config-file configs/${BENCHMARK}/t2.yaml \
                                MODEL.WEIGHTS output/${BASELINE}/${BENCHMARK}/t1/model_0019999.pth \
                                OUTPUT_DIR output/${BASELINE}/${BENCHMARK}/t2/
  
  CUDA_VISIBLE_DEVICES=${GPUS} python discover_unknown.py \
                                --config-file configs/${BENCHMARK}/t2_ft.yaml \
                                --input-txt datasets/ImageSets/Main/${BENCHMARK}/t2_ft.txt \
                                --task ${BENCHMARK}/t2_ft \
                                --output output/${BASELINE}/${BENCHMARK}/t2/unknown_rois.json \
                                --opts MODEL.WEIGHTS output/${BASELINE}/${BENCHMARK}/t2/model_0014999.pth DISCOVER_UNKNOWN True 

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                                --num-gpus 4 \
                                --dist-url tcp://127.0.0.1:${PORT} \
                                --task ${BENCHMARK}/t2_ft \
                                --config-file configs/${BENCHMARK}/t2_ft.yaml \
                                --resume \
                                MODEL.WEIGHTS output/${BASELINE}/${BENCHMARK}/t2/model_0014999.pth \
                                DISCOVER_STORE_PATH output/${BASELINE}/${BENCHMARK}/t2/unknown_rois.json \
                                OUTPUT_DIR output/${BASELINE}/${BENCHMARK}/t2/

  # CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml MODEL.WEIGHTS output/${BENCHMARK}/t2_ft/model_0014999.pth

  # CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml MODEL.WEIGHTS output/${BENCHMARK}/t3/model_0014999.pth

  # CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml MODEL.WEIGHTS output/${BENCHMARK}/t3_ft/model_0014999.pth

  # CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml MODEL.WEIGHTS output/${BENCHMARK}/t4/model_0014999.pth
else
  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml MODEL.WEIGHTS output/${BENCHMARK}/t1/model_0039999.pth

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml MODEL.WEIGHTS output/${BENCHMARK}/t2/model_0014999.pth

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml MODEL.WEIGHTS output/${BENCHMARK}/t2_ft/model_0014999.pth

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml MODEL.WEIGHTS output/${BENCHMARK}/t3/model_0014999.pth

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml MODEL.WEIGHTS output/${BENCHMARK}/t3_ft/model_0014999.pth

  CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml MODEL.WEIGHTS output/${BENCHMARK}/t4/model_0014999.pth
fi
