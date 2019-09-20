import torch.nn as nn

from blocks.encoders import RNNEncoder
from blocks.decoders import RNNDecoder
from blocks.attention import GlobalAttention
from blocks.embeddings import Embedding
from blocks.bridges import *
from blocks.DAT import DATLayer, GradientReversalLayer

class DAT_ASRModel(nn.Module):
    """
    ASR model augmented with DAT module.
    """
    def __init__(self, opts):
        super(DAT_ASRModel, self).__init__()
        self.encoder = RNNEncoder(
            rnn_type=opts.enc_rnn_type, 
            enc_layers=opts.enc_layers,
            brnn=opts.brnn,
            input_size=opts.input_size,
            enc_rnn_size=opts.enc_rnn_size, 
            enc_pooling=opts.enc_pooling, 
            dropout=opts.enc_dropout
        )
        self.W = nn.Linear(opts.enc_rnn_size, opts.dec_rnn_size, bias=False)

        if opts.bridge_type == 'copy':
            num_directions = 2 if opts.brnn else 1
            assert opts.enc_rnn_size * num_directions == opts.dec_rnn_size
            assert opts.enc_layers == opts.dec_layers
            self.bridge = lambda state: state
        
        elif opts.bridge_type == 'mlp':
            self.bridge = DenseBridge(
                enc_layers=opts.enc_layers, 
                enc_rnn_size=opts.enc_rnn_size,
                bidirectional=opts.brnn,
                dec_rnn_type=opts.dec_rnn_type, 
                dec_layers=opts.dec_layers, 
                dec_rnn_size=opts.dec_rnn_size
            )

        elif opts.bridge_type == 'zero':
            self.bridge = ZeroBridge(
                dec_rnn_type=opts.dec_rnn_type, 
                dec_layers=opts.dec_layers, 
                dec_rnn_size=opts.dec_rnn_size
            )

        self.attention = GlobalAttention(
            input_size=opts.dec_rnn_size,
            attention_type=opts.attention_type
        )
        self.generator = nn.Sequential(
            nn.Linear(opts.dec_rnn_size, opts.vocab_size, bias=False),
            nn.LogSoftmax(dim=-1)
        )
        self.embeddings = Embedding(
            word_vec_size=opts.embedding_size,
            word_vocab_size=opts.vocab_size,
            word_padding_idx=opts.padding_idx,
        )

        if opts.share_dec_weights:
            assert opts.embedding_size == opts.dec_rnn_size, \
                "Embedding and decoder state sizes must match for weight sharing."
            self.generator[0].weight = self.embeddings.word_lut.weight

        self.decoder = RNNDecoder(
            rnn_type=opts.dec_rnn_type,
            num_layers=opts.dec_layers,
            hidden_size=opts.dec_rnn_size,
            dropout=opts.dec_dropout,
            embeddings=self.embeddings,
            attention=self.attention,
            context_gate=None,
            generator=self.generator,
            sched_sampling_rate=opts.init_sched_sampling_rate
        )

        self.dat_module = nn.Sequential(
            GradientReversalLayer(
                lmbda=opts.grad_reverse_lambda),
            DATLayer(
                input_size=opts.enc_rnn_size, 
                hidden_size=opts.agg_rnn_size,
                n_layers=opts.agg_layers,
                n_accents=len(opts.accents.split(','))
            )
        )

    def forward(self, src, tgt, lengths, bptt=False):
        # Encode input sequence
        encoder_final, memory_bank, new_lengths = self.encoder(src, lengths)
        # Convert encoder state to decoder state
        encoder_final = self.bridge(encoder_final)
        # Make size of memory bank same as that of decoder
        memory_bank_dec = self.W(memory_bank)
        # Initialize decoder state if tgt is not truncated
        # F-prop via the DAT module only this time
        dat_out = None
        if bptt is False:
            self.decoder.init_state(encoder_final)
            dat_out = self.dat_module(memory_bank, new_lengths)
        # Decode sequence
        dec_out, attns = self.decoder(tgt, memory_bank_dec, new_lengths)
        return dec_out, attns, dat_out

    def update_dropout(self, p):
        self.encoder.update_dropout(p)
        self.decoder.update_dropout(p)

    def update_sched_sampling_rate(self, new_rate):
        self.decoder.sched_sampling_rate = new_rate