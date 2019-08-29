""" Optimizers class """
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import operator
import functools
from copy import copy
from math import sqrt


def build_torch_optimizer(model, opt):
    params = [p for p in model.parameters() if p.requires_grad]
    if opt.optim == 'sgd':
        optimizer = optim.SGD(
            params, 
            lr=opt.learning_rate)
    elif opt.optim == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=opt.learning_rate,
            initial_accumulator_value=opt.adagrad_accumulator_init)
    elif opt.optim == 'adadelta':
        optimizer = optim.Adadelta(
            params, 
            lr=opt.learning_rate)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(
            params,
            lr=opt.learning_rate,
            betas=[opt.adam_beta1, opt.adam_beta2],
            eps=1e-9)
    else:
        raise ValueError('Invalid optimizer type: ' + opt.optim)

    return optimizer


def make_lr_decay_fn(opt):
    if opt.decay_method == 'noam':
        return functools.partial(
            noam_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.dec_rnn_size)
    elif opt.decay_method == 'noamwd':
        return functools.partial(
            noamwd_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.dec_rnn_size,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps)
    elif opt.decay_method == 'rsqrt':
        return functools.partial(
            rsqrt_decay, warmup_steps=opt.warmup_steps)
    elif opt.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps)


def noam_decay(step, warmup_steps, model_size):
    """
    Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)))


def noamwd_decay(step, warmup_steps, model_size, 
                 rate, decay_steps, start_step=0):
    """
    Learning rate schedule optimized for huge batches
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)) *
        rate ** (max(step - start_step + decay_steps, 0) // decay_steps))


def exponential_decay(step, rate, decay_steps, start_step=0):
    """
    A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """
    Decay based on the reciprocal of the step square root.
    """
    return 1.0 / sqrt(max(step, warmup_steps))

class Optimizer(object):
    def __init__(self,
                 optimizer,
                 learning_rate,
                 learning_rate_decay_fn=None,
                 max_grad_norm=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1

    @property
    def training_step(self):
        return self._training_step

    def learning_rate(self):
        if self._learning_rate_decay_fn is None:
            return self._learning_rate
        scale = self._learning_rate_decay_fn(self._decay_step)
        return scale * self._learning_rate

    def state_dict(self):
        return {
            'training_step': self._training_step,
            'decay_step': self._decay_step,
            'optimizer': self._optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict['training_step']
        # State can be partially restored
        if 'decay_step' in state_dict:
            self._decay_step = state_dict['decay_step']
        if 'optimizer' in state_dict:
            self._optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        self._optimizer.zero_grad()

    def backward(self, loss):
        loss.backward()

    def step(self):
        learning_rate = self.learning_rate()
        for group in self._optimizer.param_groups:
            group['lr'] = learning_rate
            if self._max_grad_norm > 0:
                clip_grad_norm_(group['params'], self._max_grad_norm)
        self._optimizer.step()
        self._decay_step += 1
        self._training_step += 1