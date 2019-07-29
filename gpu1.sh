#!/usr/bin/env bash

# preprocess word level
# python preprocess.py -data_type audio -src_dir /hdddata/2scratch/datasets/speech/cv/ -train_src data/common-voice/src-train.txt -train_tgt data/common-voice/tgt-train.txt -valid_src data/common-voice/src-val.txt -valid_tgt data/common-voice/tgt-val.txt -shard_size 3000 -save_data data/common-voice/shards/word-level --overwrite --tgt_seq_length 50

# preprocess char level
# python preprocess.py -data_type audio -src_dir /hdddata/2scratch/datasets/speech/cv/ -train_src data/common-voice/src-train.txt -train_tgt data/common-voice/tgt-train.txt.converted -valid_src data/common-voice/src-val.txt -valid_tgt data/common-voice/tgt-val.txt.converted -shard_size 3000 -save_data data/common-voice/shards/char-level --overwrite --tgt_seq_length 500

# preprocess sentencepiece level - v3
# python preprocess.py -data_type audio -src_dir /home/data/MCV-v3/clips -train_src data/common-voice/us-v3/src-train.txt -train_tgt data/common-voice/us-v3/tgt-train.txt.converted -valid_src data/common-voice/us-v3/src-val.txt -valid_tgt data/common-voice/us-v3/tgt-val.txt.converted -shard_size 3000 -save_data data/common-voice/shards/us-v3/sentencepiece-level --overwrite --tgt_seq_length 1000 --src_seq_length 1000 -sample_rate 48000

# train model
CUDA_VISIBLE_DEVICES=1 python train.py -model_type audio \
 	-rnn_type LSTM \
 	-encoder_type brnn \
 	-enc_rnn_size 1024 \
 	-enc_layers 4 \
 	-audio_enc_pooling 2 \
 	-dec_rnn_size 512 \
 	-dec_layers 2 \
 	-dropout 0.5 \
 	-data data/common-voice/shards/us-v3/sentencepiece-level \
 	-save_model saved/v3_sp_drop_5_pool_2_bi_enc_1024_dec_2/model \
 	-global_attention mlp \
 	-batch_size 32 \
 	-optim adam \
 	-max_grad_norm 100 \
 	-learning_rate 0.0003 \
 	-learning_rate_decay 0.8 \
 	-train_steps 40000 \
 	-valid_steps 1000 \
 	-save_checkpoint_steps 2000 \
 	-keep_checkpoint 3 \
 	-tensorboard \
 	-tensorboard_log_dir runs/v3_sp_drop_5_pool_2_bi_enc_1024_dec_2 \
 	-gpu_ranks 0 \
	-sample_rate 48000
