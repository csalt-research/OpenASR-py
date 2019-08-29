import math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class RNNEncoder(nn.Module):
    def __init__(self, 
                 rnn_type, 
                 enc_layers,
                 brnn,
                 input_size,
                 enc_rnn_size, 
                 enc_pooling, 
                 dropout=0.0,
                 ):
        super(RNNEncoder, self).__init__()
        # type of RNN cell used
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn_type = rnn_type
        self.enc_layers = enc_layers
        self.num_directions = 2 if brnn else 1
        assert enc_rnn_size % self.num_directions == 0
        enc_rnn_size_real = enc_rnn_size // self.num_directions
        
        self.input_size = input_size

        # specifies sub-sampling rate(s) in encoder
        # different from pyramidal RNN where outputs are stacked, not pooled
        enc_pooling = enc_pooling.split(',')
        assert len(enc_pooling) == enc_layers or len(enc_pooling) == 1
        if len(enc_pooling) == 1:
            enc_pooling = enc_pooling * enc_layers
        enc_pooling = [int(p) for p in enc_pooling]
        self.enc_pooling = enc_pooling

        if type(dropout) is not list:
            dropout = [dropout]
        if max(dropout) > 0:
            self.dropout = nn.Dropout(dropout[0])
        else:
            self.dropout = None
        
        self.batchnorm_0 = nn.BatchNorm1d(enc_rnn_size, affine=True)
        self.rnn_0 = getattr(nn, rnn_type)(input_size=input_size,
                                           hidden_size=enc_rnn_size_real,
                                           num_layers=1,
                                           dropout=dropout[0],
                                           bidirectional=brnn)
        self.pool_0 = nn.MaxPool1d(enc_pooling[0])
        for l in range(enc_layers - 1):
            batchnorm = nn.BatchNorm1d(enc_rnn_size, affine=True)
            rnn = getattr(nn, rnn_type)(input_size=enc_rnn_size,
                                        hidden_size=enc_rnn_size_real,
                                        num_layers=1,
                                        dropout=dropout[0],
                                        bidirectional=brnn)
            pool = nn.MaxPool1d(enc_pooling[l + 1])
            setattr(self, 'rnn_%d' % (l + 1), rnn)
            setattr(self, 'pool_%d' % (l + 1), pool)
            setattr(self, 'batchnorm_%d' % (l + 1), batchnorm)

        self.flatten_parameters()

    def forward(self, src, lengths=None):
        batch_size, _, nfft, t = src.size()
        src = src.permute(3, 0, 2, 1).contiguous().view(t, batch_size, nfft)
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist()
        encoder_final = []

        for l in range(self.enc_layers):
            rnn = getattr(self, 'rnn_%d' % l)
            pool = getattr(self, 'pool_%d' % l)
            batchnorm = getattr(self, 'batchnorm_%d' % l)
            stride = self.enc_pooling[l]
            # Pack input sequence
            packed_emb = pack(src, lengths)
            # Unroll RNN
            memory_bank, final_state = rnn(packed_emb)
            # Store final state for this layer
            encoder_final += [final_state]
            # Unpack encoder's output sequence
            memory_bank = unpack(memory_bank)[0]
            # Sub-sample encoder's output along time dimension
            t, _, _ = memory_bank.size()
            memory_bank = memory_bank.transpose(0, 2)
            memory_bank = pool(memory_bank)
            lengths = [int(math.floor((l - stride) / stride + 1)) for l in lengths]
            memory_bank = memory_bank.transpose(0, 2)
            src = memory_bank
            t, _, num_feat = src.size()
            # Apply batch normalization
            src = batchnorm(src.contiguous().view(-1, num_feat))
            src = src.view(t, -1, num_feat)
            # Apply dropout
            if self.dropout and l + 1 != self.enc_layers:
                src = self.dropout(src)

        # Prepare final encoder state by stacking layer-wise and
        # converting to a [#layers, B, #dim x #directions] shaped tensor
        if self.rnn_type == 'LSTM':
            h_n, c_n = zip(*encoder_final)
            if self.num_directions == 2:
                h_n = torch.stack([h.transpose(0,1).view(batch_size, -1) for h in h_n])
                c_n = torch.stack([c.transpose(0,1).view(batch_size, -1) for c in c_n])
            else:
                h_n = torch.stack([h.squeeze(0) for h in h_n])
                c_n = torch.stack([c.squeeze(0) for c in c_n])
            encoder_final = (h_n.contiguous(), c_n.contiguous())
        else:
            h_n = encoder_final
            if self.num_directions == 2:
                h_n = torch.stack([h.transpose(0,1).view(batch_size, -1) for h in h_n])
            else:
                h_n = torch.stack([h.squeeze(0) for h in h_n])
            encoder_final = (h_n.contiguous(),)
        return encoder_final, memory_bank.contiguous(), orig_lengths.new_tensor(lengths)

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        for i in range(self.enc_layers - 1):
            getattr(self, 'rnn_%d' % i).dropout = dropout

    def flatten_parameters(self):
        for i in range(self.enc_layers):
            getattr(self, 'rnn_%d' % i).flatten_parameters()