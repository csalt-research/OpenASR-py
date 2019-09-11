import torch
import os
from tqdm import tqdm
import warnings
import numpy as np

from itertools import islice, cycle
from utils.logging import logger

class Vocab(object):
    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = []

        self.add('<pad>') # PAD index is 0
        self.add('<unk>') # UNK index is 1
        self.add('<bos>') # BOS index is 2
        self.add('<eos>') # EOS index is 3

    def __len__(self):
        return len(self.idx2tok)

    def add(self, token):
        if token not in self.tok2idx:
            self.tok2idx[token] = len(self.idx2tok)
            self.idx2tok.append(token)

    def encode(self, token):
        return self.tok2idx.get(token, self.tok2idx['<unk>'])

def split_corpus(path, shard_size):
    with open(path, "rb") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                shard = list(islice(f, shard_size))
                if not shard:
                    break
                yield shard

def build_vocab(src_file, max_vocab_size=0):
    with open(src_file, 'r') as f:
        tokens = f.read().split()
        freq_dict = {}
        for t in tokens:
            freq_dict[t] = freq_dict.get(t, 0) + 1
        tokens = sorted(
            list(freq_dict.items()), 
            key=lambda x: x[1], 
            reverse=True
        )
        vsize = max_vocab_size if max_vocab_size > 0 else len(tokens)
        vocab = [t[0] for t in tokens[:vsize]]
        ret = Vocab()
        for t in vocab:
            ret.add(t)
        return ret

def build_shards(src_dir, save_dir, src_file, tgt_file, vocab,
                 shard_size, feat_ext, mode='train', feats=None
                 ):
    src_shards = split_corpus(src_file, shard_size)
    tgt_shards = split_corpus(tgt_file, shard_size)

    shard_index = 0
    for src_shard, tgt_shard in zip(src_shards, tgt_shards):
        logger.info('Building %s shard %d' % (mode, shard_index))
        
        audio_feats = []
        transcriptions = []
        indices = []

        for src, tgt in tqdm(list(zip(src_shard, tgt_shard))):
            
            audio_path = os.path.join(src_dir, src.strip())
            if not os.path.exists(audio_path):
                audio_path = src.strip()
            assert os.path.exists(audio_path), \
                "audio file %s not found" % audio_path

            audio_feats.append(feat_ext(audio_path))
            transcriptions.append(tgt.strip())
            indices.append([vocab.encode(x) for x in ('<bos> '+tgt+' <eos>').split()])

        shard = {
            'src': np.asarray(audio_feats), 
            'tgt': np.asarray(transcriptions), 
            'indices': np.asarray(indices).reshape(-1,1),
            'feats': feats
        }

        shard_path = os.path.join(save_dir, '%s.%05d.pt' % (mode, shard_index))
        logger.info('Saving shard %d to %s' % (shard_index, shard_path))
        torch.save(shard, shard_path)
        shard_index += 1