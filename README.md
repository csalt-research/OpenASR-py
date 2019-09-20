# Active Adversarial Accent Adaptation in ASR

This repository contains the codebase for my BTP on 'Active Adversarial Accent Adaptation in ASR' under the supervision of Prof. Preethi Jyothi (IIT Bombay) and Dr. Maneesh Singh (Director, Verisk AI), and mentorship of Deepak Mittal (Verisk AI). It is based on OpenNMT-py.

## Installation

We recommend using `conda` for setting up the environment. After `miniconda` has been successfully installed, follow the steps below:

*Note*: `conda` gives an intermittent `HTTP 000 Connection Error`. Retrying the command, several times at worst, solves the issue.
```bash
# create environment
conda create -n aa
conda activate aa
# install pytorch 1.1 and its dependencies
conda install pytorch=1.1 -c pytorch
# clone codebase and install its dependencies
git clone https://github.com/ys1998/accent-adaptation.git
cd accent-adaptation/
pip install -r requirements.txt
# install torchaudio
pip install librosa
sudo apt-get install -y sox libsox-dev libsox-fmt-all
pip install git+https://github.com/pytorch/audio@d92de5b
# install sentencepiece
pip install sentencepiece
# install library for error rate computation
pip install python-levenshtein
```

## Generating train/val/test splits

TODO

## Preprocessing and preparing data

Perform these steps to filter and preprocess the data.
```bash
# remove long utterances that often cause OoM
python filter_data.py
# train a sentencepiece model and use it to encode transcripts
python convert_to_sp.py
# read mp3 files, compute features, sharding and saving as .pt files
python preprocess.py -data_type audio -src_dir <PATH TO DATA DIR> -train_src <PATH TO SRC TRAIN FILE> -train_tgt <PATH TO TGT TRAIN FILE> -valid_src <PATH TO SRC VALID FILE> -valid_tgt <PATH TO TGT VALID FILE> -save_data <PATH TO SAVE DIR> --src_seq_length <MAX SRC SEQ LEN> --tgt_seq_length <MAX TGT SEQ LEN> -sample_rate <SAMPLE RATE> -shard_size <SHARD SIZE> [--overwrite]
```

## Training
In order to train the baseline ASR model, use this command
```bash
python train.py -model_type audio \
  	-rnn_type LSTM \
  	-encoder_type brnn \
  	-enc_rnn_size 1024 \
  	-enc_layers 4 \
  	-audio_enc_pooling 2 \
  	-dec_rnn_size 512 \
  	-dec_layers 1 \
  	-dropout 0.5 \
  	-data <PATH WHERE SHARDS ARE STORED> \
  	-save_model <PATH TO SAVE DIR> \
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
  	-tensorboard_log_dir <PATH TO LOG DIR> \
  	-gpu_ranks 0 \
 	-sample_rate 48000
```

## Computing ERs
```bash
bash compute_er.sh
```
