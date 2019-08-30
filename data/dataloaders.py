import torch
import glob
import os
from utils.logging import logger
import numpy as np
import random
from itertools import islice

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

"""
Simple dataloader to cycle through a list of shard paths.
"""
class ShardedDataLoader(object):
    def __init__(self,
                 shard_root_dir,
                 batch_size,
                 bucket_size,
                 padding_idx=0,
                 mode='train',
                 repeat=True
                 ):
        # Collect all shard paths
        shard_paths = glob.glob(
            '%s/**/%s.*.pt' % (shard_root_dir, mode), 
            recursive=True)
        self.shard_paths = sorted(shard_paths)

        # Store parameters
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.padding_idx = padding_idx
        self.mode = mode
        self.repeat = repeat

        # Load first shard
        self.pointer = 0
        self.load_next_shard()

    def load_next_shard(self):
        path = self.shard_paths[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.shard_paths)

        self.shard = torch.load(path)
        n_examples = len(self.shard['src'])
        idx = np.arange(n_examples).tolist()
        self.order = np.concatenate([random.sample(x, len(x)) for x in chunks(idx, self.bucket_size)])
        logger.info('Loaded %d examples from %s' % (n_examples, path))

    def pad(self, lst, max_len=None, sort=False):
        """
        Pad given list of sequences.
            lst     - list of sequences with shape [T x D]
            max_len - length upto which each sequence is to be padded
            sort    - whether to sort by descending order of lengths
        """
        lengths = np.asarray([x.shape[0] for x in lst])
        T = max_len if max_len is not None else max(lengths)
        B = len(lst)
        D = lst[0].shape[1]
        ret = np.full([T, B, D], self.padding_idx)
        for i, x in enumerate(lst):
            ret[:x.shape[0], i, :] = x
        if sort:
            order = np.argsort(lengths)[::-1]
            return ret[:, order, :], lengths[order], order
        else:
            return ret, lengths

    def __iter__(self):
        while True:
            current_batch = self.order[:self.batch_size]
            self.order = self.order[self.batch_size:]

            src = self.shard['src'][current_batch]
            tgt = self.shard['indices'][current_batch]
            feat = self.shard['feats'] 

            src, src_len, idx = self.pad(src, sort=True)
            tgt, tgt_len = self.pad(tgt)
            tgt, tgt_len = tgt[:, idx], tgt_len[idx]
            
            # print(src.shape, src_len.shape, tgt.shape, tgt_len.shape)

            yield torch.FloatTensor(src), torch.LongTensor(src_len), \
                torch.LongTensor(tgt), torch.LongTensor(tgt_len), feat

            if self.order.size == 0:
                if self.pointer == 0 and not self.repeat:
                    self.load_next_shard()
                    break
                else:
                    self.load_next_shard()