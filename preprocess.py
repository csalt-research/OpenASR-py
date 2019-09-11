import torch, os
from configargparse import ArgumentParser
from utils.logging import logger
from utils.opts import build_preprocess_parser
from utils.misc import ensure_dir
from data.data_utils import build_vocab, build_shards
from data.features import AudioFeatureExtractor

ACCENTS = {'us': 0, 'england': 1, 'indian': 2, 'australia': 3}

def preprocess(opts):
    # Create dirs if not exist
    ensure_dir(opts.save_dir)

    # Build vocabulary
    logger.info('Building vocabulary from %s' % opts.vocab)
    vocab = build_vocab(opts.vocab, opts.max_vocab_size)
    logger.info('Saving vocabulary of size %d to %s' % \
        (len(vocab), os.path.join(opts.save_dir, 'vocab.pt')))
    torch.save(vocab, os.path.join(opts.save_dir, 'vocab.pt'))

    # Build feature extractor
    feat_ext = AudioFeatureExtractor(
        sample_rate=opts.sample_rate, 
        window_size=opts.window_size, 
        window_stride=opts.window_stride,
        window=opts.window,
        feat_type=opts.feat_type, 
        normalize_audio=opts.normalize_audio)
    torch.save(feat_ext, os.path.join(opts.save_dir, 'feat_ext.pt'))

    # Build train shards
    for src_train, tgt_train in zip(opts.src_train, opts.tgt_train):
       accent = src_train.split('/')[-2]
       feats = {'accent': ACCENTS[accent], 'labeled': accent=='us'}
       build_shards(src_dir=opts.src_dir, 
                    save_dir=os.path.join(opts.save_dir, accent), 
                    src_file=src_train, 
                    tgt_file=tgt_train,
                    vocab=vocab,
                    shard_size=opts.shard_size,
                    feat_ext=feat_ext, 
                    mode='train', 
                    feats=feats)

    # Build validation shards
    for src_valid, tgt_valid in zip(opts.src_valid, opts.tgt_valid):
        accent = src_valid.split('/')[-2]
        feats = {'accent': ACCENTS[accent], 'labeled': True}
        build_shards(src_dir=opts.src_dir, 
                     save_dir=os.path.join(opts.save_dir, accent), 
                     src_file=src_valid, 
                     tgt_file=tgt_valid,
                     vocab=vocab,
                     shard_size=opts.shard_size,
                     feat_ext=feat_ext, 
                     mode='valid', 
                     feats=feats)

if __name__ == '__main__':
    parser = ArgumentParser(description='preprocess.py')
    parser = build_preprocess_parser(parser)
    opts = parser.parse_args()
    preprocess(opts)
