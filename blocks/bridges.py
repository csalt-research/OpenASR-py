import torch
import torch.nn as nn

class ZeroBridge(nn.Module):
    def __init__(self, 
                 dec_rnn_type, 
                 dec_layers, 
                 dec_rnn_size
                 ):
        super(ZeroBridge, self).__init__()
        self.dec_rnn_type = dec_rnn_type
        self.dec_layers = dec_layers
        self.dec_rnn_size = dec_rnn_size

    def forward(self, encoder_final):
        if self.dec_rnn_type == 'GRU':
            batch_size = encoder_final[0].size(1)
            return (torch.zeros(self.dec_layers, batch_size, self.dec_rnn_size),)
        else:
            batch_size = encoder_final[0].size(1)
            return (torch.zeros(self.dec_layers, batch_size, self.dec_rnn_size), \
                    torch.zeros(self.dec_layers, batch_size, self.dec_rnn_size))

class DenseBridge(nn.Module):
    def __init__(self, 
                 enc_layers, 
                 enc_rnn_size,
                 bidirectional,
                 dec_rnn_type, 
                 dec_layers, 
                 dec_rnn_size
                 ):
        super(DenseBridge, self).__init__()
        self.dec_rnn_type = dec_rnn_type
        self.dec_layers = dec_layers
        self.dec_rnn_size = dec_rnn_size

        num_directions = 2 if bidirectional else 1
        self.linear_h = nn.Linear(
            enc_layers * enc_rnn_size * num_directions,
            dec_layers * dec_rnn_size,
            bias=True
        )
        if self.dec_rnn_type == 'LSTM':
            self.linear_c = nn.Linear(
                enc_layers * enc_rnn_size * num_directions,
                dec_layers * dec_rnn_size,
                bias=True
            )

    def forward(self, encoder_final):
        h_enc = encoder_final[0]
        batch_size = h_enc.size(1)
        h_dec = self.linear_h(h_enc.transpose(0,1).view(batch_size, -1))
        h_dec = h_dec.view(batch_size, self.dec_layers, self.dec_rnn_size).transpose(0,1)

        if self.dec_rnn_type == 'GRU':
            return (h_dec, )
        else:
            if len(encoder_final) == 1:
                c_dec = self.linear_c(h_enc.transpose(0,1).view(batch_size, -1))
            else:
                c_enc = encoder_final[1]
                c_dec = self.linear_c(c_enc.transpose(0,1).view(batch_size, -1))
            c_dec = c_dec.view(batch_size, self.dec_layers, self.dec_rnn_size).transpose(0,1)
            return (h_dec, c_dec)