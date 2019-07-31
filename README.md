# Active Adversarial Accent Adaptation in ASR

This repository contains the codebase for my BTP on 'Active Adversarial Accent Adaptation in ASR' under the supervision of Prof. Preethi Jyothi, IIT Bombay and Dr. Maneesh Singh, Director - Verisk AI, and mentorship of Deepak Mittal, Verisk AI. It is based on OpenNMT-py.

## Installation

We recommend using `conda` for setting up the environment. After `miniconda` has been successfully installed, follow the steps below:

*_Note_*: `conda` gives an intermittent `HTTP 000 Connection Error`. Retrying the command, several times at worst, solves the issue.
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

## Data preparation

Perform these steps to filter and preprocess the data.
```bash
# remove long utterances that often cause OoM
python filter_data.py
# train a sentencepiece model and use it to encode transcripts
python convert_to_sp.py
# read mp3 files, compute features, sharding and saving as .pt files
python preprocess.py
```

## Training
```bash
python train.py
```

## Computing ERs
```bash
bash compute_er.sh
```
