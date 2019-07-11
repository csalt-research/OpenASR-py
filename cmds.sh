#!/usr/bin/env bash

# preprocess word level
# python preprocess.py -data_type audio -src_dir /hdddata/2scratch/datasets/speech/cv/ -train_src data/common-voice/src-train.txt -train_tgt data/common-voice/tgt-train.txt -valid_src data/common-voice/src-val.txt -valid_tgt data/common-voice/tgt-val.txt -shard_size 3000 -save_data data/common-voice/shards/word-level --overwrite --tgt_seq_length 50

# preprocess char level
# python preprocess.py -data_type audio -src_dir /hdddata/2scratch/datasets/speech/cv/ -train_src data/common-voice/src-train.txt -train_tgt data/common-voice/tgt-train.txt.converted -valid_src data/common-voice/src-val.txt -valid_tgt data/common-voice/tgt-val.txt.converted -shard_size 3000 -save_data data/common-voice/shards/char-level --overwrite --tgt_seq_length 500

python preprocess.py -data_type audio -src_dir /hdddata/2scratch/datasets/speech/common-voice/clips -train_src data/common-voice/us/src-train.txt -train_tgt data/common-voice/us/tgt-train.txt.converted -valid_src data/common-voice/us/src-val.txt -valid_tgt data/common-voice/us/tgt-val.txt.converted -shard_size 3000 -save_data data/common-voice/us/shards/sentencepiece-level --overwrite --tgt_seq_length 100 -sample_rate 48000

# train model
CUDA_VISIBLE_DEVICES=0 python train.py -model_type audio \
	-rnn_type LSTM \
	-encoder_type brnn \
	-enc_rnn_size 1024 \
	-enc_layers 4 \
	-audio_enc_pooling 2 \
	-dec_rnn_size 512 \
	-dec_layers 1 \
	-dropout 0.5 \
	-data data/common-voice/us/shards/sentencepiece-level \
	-save_model saved/sp_drop_5_pool_2_bi_enc_1024/model \
	-global_attention mlp \
	-batch_size 64 \
	-optim adam \
	-max_grad_norm 100 \
	-learning_rate 0.0003 \
	-learning_rate_decay 0.8 \
	-train_steps 20000 \
	-valid_steps 500 \
	-save_checkpoint_steps 1000 \
	-keep_checkpoint 3 \
	-tensorboard \
	-tensorboard_log_dir runs/sp_drop_5_pool_2_bi_enc_1024 \
	-gpu_ranks 0
