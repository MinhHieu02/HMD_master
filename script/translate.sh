#!/usr/bin/env bash

#PROJ=/home/evanyfgao/Distractor-Generation-RACE

export CUDA_VISIBLE_DEVICES=0
FULL_MODEL_NAME=$1

python3 -u ../translate.py \
    -model=../data/model/${FULL_MODEL_NAME}.pt \
    -data=../data/feature_data/race_test_large_features.json \
    -output=../data/pred/${FULL_MODEL_NAME}.txt \
    -share_vocab \
    -block_ngram_repeat=1 \
    -replace_unk \
    -batch_size=1 \
    -beam_size=10\
    -n_best=10 \
    -gpuid=-1 \
    -report_eval_every=10 \
    #-feat_name=pos_ner_dep_lemma \
    #-n_feats=4 \
    #-max_test_sentences=5
    #-log_file=../logs/translate.log
