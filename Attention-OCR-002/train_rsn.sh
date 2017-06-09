#!/usr/bin/env bash
# export CUDA_VISIBLE_DIVICES=0
py=/export/App/anaconda-tf-1.0/bin/python

$py src/launcher.py \
	--phase=train \
	--data-path=dataset/CP_3000/annotation_train_words.txt \
	--data-base-dir=dataset/CP_3000 \
	--log-path=log_01_16.log \
	--attn-num-hidden 800 \
	--batch-size 64 \
	--model-dir=model_new_800  \
	--initial-learning-rate=1.0 \
	--load-model  \
	--num-epoch=300000 \
       	--gpu-id=0 \
       	# --use-gru \
	--steps-per-checkpoint=2000 \
        --target-embedding-size=20
