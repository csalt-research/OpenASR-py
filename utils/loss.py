import torch
import torch.nn as nn
import torch.nn.functional as F

class ShardedCELoss(nn.Module):
    def __init__(self, padding_idx=0):
        super(ShardedCELoss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    def _make_shard_state(self, target, output, range_):
        return {"output": output, "target": target[range_[0] + 1: range_[1], :, 0]}

    def _compute_loss(self, target, output):
        scores = output.view(-1)
        gtruth = target.view(-1)
        loss = self.criterion(scores, gtruth)
        return loss

    def __call__(self,
                 target,
                 output,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        if trunc_size is None:
            trunc_size = target.size(0) - trunc_start
        
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(target, output, trunc_range)
        
        if shard_size == 0:
            loss = self._compute_loss(target, **shard_state)
            return loss / float(normalization), True

        total_loss = 0.0
        for shard in shards(shard_state, shard_size):
            loss = self._compute_loss(target, **shard)
            loss.div(float(normalization)).backward()
            total_loss += loss.item()
        return total_loss, False

def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)

def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)