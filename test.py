import os
import torch
from configargparse import ArgumentParser

from data.dataloaders import ShardedDataLoader

from utils.logging import logger
from utils.opts import build_test_parser
from utils.decode import beam_decode
from utils.misc import edit_distance

from models.ASR import ASRModel

def evaluate(model, dataloader, vocab, opts):
    # Identify device to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set model in evaluation mode.
    model.eval()
    model.to(device)

    

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
        batch_size=opts.batch_size,
        bucket_size=1,
        padding_idx=0,
        mode=opts.split,
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
    logger.info('Evaluating performance on \'%d\' split' % opts.split)
    test_loss, test_er = evaluate(model, test_dataloader, vocab, opts)
    logger.info('Avg. NLL %.4f, Avg. ER %.2f' % (test_loss, 100.0*test_er))
    ################################################################################

if __name__ == "__main__":
    parser = ArgumentParser(description='test.py')
    parser = build_test_parser(parser)
    opts = parser.parse_args()
    main(opts)