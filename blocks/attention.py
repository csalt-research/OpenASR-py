import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import sequence_mask

class GlobalAttention(nn.Module):
    def __init__(self, input_size, attention_type="dot"):
        super(GlobalAttention, self).__init__()
        assert attention_type in ["dot", "general", "mlp"]

        self.input_size = input_size
        self.attention_type = attention_type
        dim = self.input_size

        if attention_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
            self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        elif attention_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
            self.linear_out = nn.Linear(dim * 2, dim, bias=True)        

    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        assert src_batch == tgt_batch
        assert src_dim == tgt_dim
        assert self.input_size == src_dim

        if self.attention_type in ["general", "dot"]:
            if self.attention_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.input_size
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None):
        # Whether input is provided one step at a time or not
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        assert batch == batch_
        assert dim == dim_
        assert self.input_size == dim

        # Compute attention scores
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # make it broadcastable
            align.masked_fill_(1 - mask, -float('inf'))

        # Normalize attention weights
        align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # Generate context vector c_t as the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # Concatenate context vector with source
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            assert batch == batch_
            assert dim == dim_
            batch_, source_l_ = align_vectors.size()
            assert batch == batch_
            assert source_l == source_l_

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            assert target_l == target_l_
            assert batch == batch_
            assert dim == dim_
            target_l_, batch_, source_l_ = align_vectors.size()
            assert target_l == target_l_
            assert batch == batch_
            assert source_l == source_l_

        return attn_h, align_vectors
