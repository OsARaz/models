#!/bin/bash

set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/fresh_train_inception_v1_flowers

# Where the dataset is saved to.
DATASET_DIR=/tmp/flowers

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --preprocessing_name=inception_v1 \
  --max_number_of_steps=2000 \
  --batch_size=32 \
  --save_interval_secs=30 \
  --save_summaries_secs=30 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=5 \
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1
