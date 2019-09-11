import torch
import torch.nn as nn
import random
from blocks.custom_rnn import StackedGRU, StackedLSTM

class RNNDecoder(nn.Module):
    def __init__(self, 
                 rnn_type,
                 num_layers,
                 hidden_size,
                 embeddings,
                 generator,
                 dropout=0.0,
                 attention=None,
                 context_gate=None,
                 sched_sampling_rate=0.0
                 ):
        super(RNNDecoder, self).__init__()

        # RNN parameters
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = embeddings.embedding_size + hidden_size
        rnn_class = StackedLSTM if rnn_type == 'LSTM' else StackedGRU
        self.rnn = rnn_class(input_size=self.input_size,
                             rnn_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)
        
        # Decoder state
        self.state = {}
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Scheduled sampling
        self.sched_sampling_rate = sched_sampling_rate

        # External modules
        self.embeddings = embeddings
        self.attention = attention
        self.context_gate = context_gate
        self.generator = generator

        self.rnn.flatten_parameters()

    def init_state(self, encoder_final):
        batch_size = encoder_final[0].size(1)
        device = encoder_final[0].device
        self.state["hidden"] = encoder_final
        self.state["input_feed"] = encoder_final[0].data \
            .new_zeros((batch_size, self.hidden_size)) \
            .unsqueeze(0).to(device)

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None):
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        
        if tgt.dim() == 2:
            decoding = True
            tgt_batch = 1
            tgt = tgt.unsqueeze(0)
        else:
            t, tgt_batch, _ = tgt.size()
            decoding = False

        assert input_feed_batch == tgt_batch

        dec_outs, attns = [], []

        dec_state = self.state["hidden"]
        current_input = self.embeddings(tgt[0].unsqueeze(0)).squeeze(0)

        # Input feed concatenates the `hidden state` with input at every time
        # step when attention module is absent, and the `context vector` when
        # attention module is present.

        # Whether to use teacher forcing or not
        use_teacher_forcing = random.random() > self.sched_sampling_rate

        # Iterate over timesteps
        max_time = tgt.size(0) if not decoding else 2
        for t in range(1, max_time):
            decoder_input = torch.cat([current_input, input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)

            if self.attention is not None:
                decoder_output, p_attn = self.attention(
                    rnn_output,
                    memory_bank.transpose(0, 1),
                    memory_lengths=memory_lengths)
                attns += [p_attn]
            else:
                decoder_output = rnn_output
            
            if self.context_gate is not None:
                decoder_output = self.context_gate(
                    decoder_input, 
                    rnn_output, 
                    decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            # Generate scores/probability distribution over vocabulary
            scores = self.generator(decoder_output)
            dec_outs += [scores]

            if use_teacher_forcing and not decoding:
                current_input = self.embeddings(tgt[t].unsqueeze(0)).squeeze(0)
            else:
                _, idx = torch.max(scores, dim=1)
                idx = idx.view(1, -1, 1)
                current_input = self.embeddings(idx).squeeze(0)

        # Update state variables
        self.state["hidden"] = dec_state
        self.state["input_feed"] = input_feed.unsqueeze(0)

        return torch.stack(dec_outs), torch.stack(attns)

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)