import torch
import torch.nn as nn
import os
from .logging import logger

class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def save_checkpoint(save_dir, step, model, optimizer, opts):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'opts': opts
    }
    save_path = os.path.join(save_dir, 'checkpoint.%d.pt' % step)
    logger.info('Saving checkpoint to %s' % save_path)
    torch.save(checkpoint, save_path)
    return save_path

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]

def ensure_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        return
    else:
        os.makedirs(path)
        return
