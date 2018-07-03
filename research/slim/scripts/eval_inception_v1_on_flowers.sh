#!/bin/bash
# Evaluates the model on the Flowers validation set.

set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/fresh_train_inception_v1_flowers

# Where the dataset is saved to.
DATASET_DIR=/tmp/flowers

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1
