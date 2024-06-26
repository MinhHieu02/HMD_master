 python3 -u ../preprocess.py \
         -train_dir=../data/feature_data/race_train_large_features.json \
         -valid_dir=../data/feature_data/race_dev_large_features.json \
         -save_data=../data/processed \
         -share_vocab \
         -total_token_length=550 \
         -src_seq_length=60 \
         -src_sent_length=40 \
         -lower \
         -feat_name=pos_ner_dep_lemma \
         -src_vocab_size=100000 \
         -tgt_vocab_size=100000 \
         -log_file=../logs/preprocess.log

python3 ../embeddings_to_torch.py \
       -output_file=../data/processed.bert \
       -dict_file=../data/processed.vocab.pt \
       -verbose
        