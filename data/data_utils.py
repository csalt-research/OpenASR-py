import torch
import os
from tqdm import tqdm
import numpy as np
from multiprocessing.pool import Pool

from itertools import islice, cycle
from utils.logging import logger
from utils.misc import ensure_dir

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

    def decode(self, token_id):
        assert token_id < len(self.idx2tok), \
            'token id must be less than %d, got %d' % (len(self.idx2tok), token_id)
        return self.idx2tok[token_id]

def split_corpus(path, shard_size):
    with open(path, "r") as f:
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

def _worker(args):
    src, tgt, feat_ext, vocab = args
    if tgt == '':
        return None
    return feat_ext(src), tgt, [vocab.encode(x) for x in ('<bos> '+tgt+' <eos>').split()]

def build_shards(src_dir, save_dir, src_file, tgt_file, vocab,
                 shard_size, feat_ext, mode='train', feats=None
                 ):
    src_shards = split_corpus(src_file, shard_size)
    tgt_shards = split_corpus(tgt_file, shard_size)
    ensure_dir(save_dir)

    shard_index = 0
    for src_shard, tgt_shard in zip(src_shards, tgt_shards):
        logger.info('Building %s shard %d' % (mode, shard_index))

        audio_paths = [os.path.join(src_dir, p.strip()) for p in src_shard]
        assert all([os.path.exists(p) for p in audio_paths]), \
            "following audio files not found: %s" % \
            ' '.join([p.strip() for p in audio_paths if not os.path.exists(p)])
        targets = [t.strip() for t in tgt_shard]

        src_tgt_pairs = list(zip(audio_paths, targets, cycle([feat_ext]), cycle([vocab])))

        with Pool(50) as p:
            result = list(tqdm(p.imap(_worker, src_tgt_pairs), total=len(src_tgt_pairs)))
            result = [r for r in result if r is not None]
            audio_feats, transcriptions, indices = zip(*result)

        shard = {
            'src': np.asarray(audio_feats), 
            'tgt': np.asarray(transcriptions), 
            'indices': np.asarray([np.asarray(x).reshape(-1,1) for x in indices]),
            'feats': feats
        }

        shard_path = os.path.join(save_dir, '%s.%05d.pt' % (mode, shard_index))
        logger.info('Saving shard %d to %s' % (shard_index, shard_path))
        torch.save(shard, shard_path)
        shard_index += 1