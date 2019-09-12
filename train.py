import os, time
import torch
from torch.nn.init import xavier_uniform_
from configargparse import ArgumentParser

from data.dataloaders import ShardedDataLoader

from utils.logging import logger
from utils.opts import build_train_parser
from utils.loss import ShardedCELoss
from utils.optimizers import Optimizer, build_torch_optimizer, make_lr_decay_fn
from utils.misc import save_checkpoint, set_random_seed

from models.ASR import ASRModel
from .test import evaluate

def maybe_update_dropout(model, step):
    pass

def main(opts):
    # Load vocabulary and feature extractor
    vocab = torch.load(os.path.join(opts.data, 'vocab.pt'))
    logger.info('Loaded vocabulary from %s, size %d {%s}' % \
        (os.path.join(opts.data, 'vocab.pt'), len(vocab), ', '.join(vocab.idx2tok)))
    # Add vocabulary arguments to opts
    opts.vocab_size = len(vocab)
    opts.padding_idx = vocab.encode('<pad>')

    feat_ext = torch.load(os.path.join(opts.data, 'feat_ext.pt'))
    # Add feat arguments to opts
    opts.input_size = feat_ext.feat_dim

    # Identify device to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build data loader(s)
    train_dataloader = ShardedDataLoader(
        shard_root_dir=opts.data,
        batch_size=opts.train_batch_size,
        bucket_size=opts.bucket_size,
        padding_idx=0,
        mode='train',
        repeat=not opts.single_pass)

    valid_dataloader = ShardedDataLoader(
        shard_root_dir=opts.data,
        batch_size=opts.eval_batch_size,
        bucket_size=1,
        padding_idx=0,
        mode='valid',
        repeat=False)

    # Define objective
    criterion = ShardedCELoss()

    # TensorboardX writer
    if opts.tensorboard_dir:
        ensure_dir(opts.tensorboard_dir)
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(opts.tensorboard_dir)

    # Build model
    if opts.checkpoint and os.path.exists(opts.checkpoint):
        # Load checkpoint if we resume from a previous training
        logger.info('Loading checkpoint from %s' % opts.checkpoint)
        checkpoint = torch.load(opts.checkpoint, \
            map_location=lambda storage, loc: storage)
        ckpt_opts = checkpoint['opts']

        # Build model and optimizer
        model = ASRModel(ckpt_opts)
        model.load_state_dict(checkpoint['model'], strict=False)

        optimizer = Optimizer(
            optimizer=build_torch_optimizer(model, ckpt_opts),
            learning_rate=ckpt_opts.learning_rate,
            learning_rate_decay_fn=make_lr_decay_fn(ckpt_opts),
            max_grad_norm=ckpt_opts.max_grad_norm)
        optimizer.load_state_dict(checkpoint['optim'])

    else:
        logger.info('Building model')
        model = ASRModel(opts)
        if opts.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-opts.param_init, opts.param_init)
        if opts.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        logger.info('Building optimizer')
        optimizer = Optimizer(
            optimizer=build_torch_optimizer(model, opts),
            learning_rate=opts.learning_rate,
            learning_rate_decay_fn=make_lr_decay_fn(opts),
            max_grad_norm=opts.max_grad_norm)

    logger.info(model)
    logger.info('#Parameters: %d' % sum([p.nelement() for p in model.parameters()]))

    ################################################################################
    best_val_er = None
    train_loss = 0.0
    start_time = time.time()
    model.to(device)

    for i, batch in enumerate(train_dataloader):
        model.train()
        src, src_len, tgt, tgt_len, feat = batch
        step = optimizer.training_step

        # Copy to device
        src = src.to(device).contiguous()
        src_len = src_len.to(device).contiguous()
        tgt = tgt.to(device).contiguous()
        tgt_len = tgt_len.to(device).contiguous()

        # Update dropout
        maybe_update_dropout(model, step)

        # Perform BPTT
        bptt = False
        trunc_size = opts.bptt if opts.bptt > 0 else tgt.size(0)
        normalization = (tgt_len - 1).sum().item()
        # normalization = opts.train_batch_size

        tgt_in = tgt[:-1]
        for j in range(0, tgt_in.size(0), trunc_size):
            # Find truncated target
            tgt_trunc = tgt_in[j : j+trunc_size]
            # Clear accumulated gradients
            optimizer.zero_grad()
            # F-prop model
            dec_outs, attns = model(src, tgt_trunc, src_len, bptt=bptt)
            # Enable BPTT for next chunks
            bptt = True
            # Compute loss
            loss, bprop = criterion(
                target=tgt,
                output=dec_outs,
                normalization=normalization,
                shard_size=opts.shard_size,
                trunc_start=j,
                trunc_size=trunc_size)
            # B-prop loss and accumulate gradients
            if bprop:
                optimizer.backward(loss)
            optimizer.step()
            # Do not b-prop fully when truncated
            if model.decoder.state is not None:
                model.decoder.detach_state()
            # Update statistics
            train_loss += loss.item()

        # Print status
        if i>0 and i % opts.log_every == 0:
            avg_nll = train_loss/opts.log_every
            avg_time = (time.time() - start_time)/opts.log_every
            logger.info(' * Batches %d/%d Avg. NLL %.4f Time %.2fs/batch' % \
                (i, opts.train_steps, avg_nll, avg_time))

            if opts.tensorboard_dir:
                tb_writer.add_scalar('train/nll', avg_nll, i)
                tb_writer.add_scalar('train/time', avg_time, i)
            
            train_loss = 0.0
            start_time = time.time()

        # Validation
        if i>0 and i % opts.eval_steps == 0:
            logger.info('Validating performance')
            val_loss, val_er = evaluate(model, valid_dataloader, vocab, opts)
            logger.info(' * Avg. NLL %.4f' % val_loss)
            logger.info(' * Avg. ER %.2f' % val_er['ER'])
            logger.info(' * Avg. WER %.2f' % val_er['WER'])
            logger.info(' * Avg. CER %.2f' % val_er['CER'])

            if best_val_er is None or val_er < best_val_er:
                best_val_er = val_er
                p = save_checkpoint(opts.save_dir, step, model, optimizer, opts)

            if opts.tensorboard_dir:
                tb_writer.add_scalar('val/nll', val_loss, i)
                tb_writer.add_scalar('val/wer', val_er['WER'], i)
                tb_writer.add_scalar('val/cer', val_er['CER'], i)
                tb_writer.add_scalar('val/er', val_er['ER'], i)

            start_time = time.time()

        # Termination condition
        if not opts.single_pass and i >= opts.train_steps:
            break

if __name__ == "__main__":
    parser = ArgumentParser(description='train.py')
    parser = build_train_parser(parser)
    opts = parser.parse_args()
    set_random_seed(opts.seed)
    main(opts)
