import sys
sys.path.insert(0, '../..')

import os
import torch
from configargparse import ArgumentParser

from data.dataloaders import ShardedDataLoader
from utils.logging import logger
from utils.misc import set_random_seed
from translate.translator import Translator
from models.ASR import ASRModel
from opts import build_test_parser

def evaluate(model, dataloader, vocab, opts):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set model in evaluation mode.
    model.eval()
    model.sched_sampling_rate = 0.0
    model.to(device)

    # Build translator
    translator = Translator(
        model,
        dataloader,
        vocab,
        device,
        n_best=opts.n_best,
        min_length=opts.min_length,
        max_length=opts.max_length,
        ratio=opts.ratio,
        beam_size=opts.beam_size,
        block_ngram_repeat=opts.block_ngram_repeat,
        ignore_when_blocking=set(opts.excluded_toks.split(',')),
        out_file=opts.out,
        verbose=opts.verbose
    )
    all_scores, all_preds, all_ers = translator.translate()

    # Compute average NLL
    best_scores = [x[0] for x in all_scores]
    avg_nll = -sum(best_scores) / float(len(best_scores))

    # Compute average ERs
    total_ed = {'ER':0, 'WER':0, 'CER':0}
    total_cnt = {'ER':0, 'WER':0, 'CER':0}
    for x in all_ers:
        for k in x.keys():
            total_ed[k] += x[k][0]
            total_cnt[k] += x[k][1]
    avg_er = {k: 100.*total_ed[k]/float(total_cnt[k]) for k in total_ed.keys()}

    return avg_nll, avg_er

def main(opts):
    # Load vocabulary and feature extractor
    vocab = torch.load(os.path.join(opts.data, 'vocab.pt'))
    logger.info('Loaded vocabulary from %s, size %d' % \
        (os.path.join(opts.data, 'vocab.pt'), len(vocab)))
    
    # Add vocabulary arguments to opts
    opts.vocab_size = len(vocab)
    opts.padding_idx = vocab.encode('<pad>')

    # Add feat arguments to opts
    feat_ext = torch.load(os.path.join(opts.data, 'feat_ext.pt'))
    opts.input_size = feat_ext.feat_dim

    # Identify device to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build data loader
    test_dataloader = ShardedDataLoader(
        shard_root_dir=opts.data,
        batch_size=opts.eval_batch_size,
        bucket_size=1,
        padding_idx=0,
        mode=opts.eval_split,
        repeat=False)

    # Load checkpoint from a saved previous training instance
    logger.info('Loading checkpoint from %s' % opts.checkpoint)
    checkpoint = torch.load(opts.checkpoint, map_location=lambda storage, loc: storage)
    ckpt_opts = checkpoint['opts']

    # Build model
    model = ASRModel(ckpt_opts)
    model.load_state_dict(checkpoint['model'], strict=False)

    logger.info(model)
    logger.info('#Parameters: %d' % sum([p.nelement() for p in model.parameters()]))

    ################################################################################
    logger.info('Evaluating performance on \'%s\' split' % opts.eval_split)
    test_loss, test_er = evaluate(model, test_dataloader, vocab, opts)
    logger.info(' * Avg. NLL %.4f' % test_loss)
    logger.info(' * Avg. ER %.2f' % test_er['ER'])
    logger.info(' * Avg. WER %.2f' % test_er['WER'])
    logger.info(' * Avg. CER %.2f' % test_er['CER'])
    ################################################################################

if __name__ == "__main__":
    parser = ArgumentParser(description='test.py')
    parser = build_test_parser(parser)
    opts = parser.parse_args()
    set_random_seed(opts.seed)
    main(opts)
