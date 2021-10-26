#!/bin/bash

CFG_NAME="instagram"

python train_net.py --num-gpus 1 --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-${CFG_NAME}.yaml OUTPUT_DIR output/${CFG_NAME} 