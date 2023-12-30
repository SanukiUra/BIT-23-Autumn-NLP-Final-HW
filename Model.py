import os
import torch.nn as nn
import torch
import torch.nn.modules.transformer as ts
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, max_len: int, vocab_size: int, n_head: int = 6, n_en: int = 6, n_de: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, d_embed: int = 300, embedding_weight=None):
        super(Transformer, self).__init__()
        self.seed = 42
        torch.manual_seed(self.seed)

        self.pos_encoder = PositionalEncoding(d_model=d_embed, max_len=max_len)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embed, _weight=embedding_weight)

        self.encoder_norm = nn.LayerNorm(d_embed)
        self.encoder_layer = ts.TransformerEncoderLayer(d_model=d_embed, nhead=n_head, dim_feedforward=d_ff,
                                                        dropout=dropout, activation='relu')
        self.encoder = ts.TransformerEncoder(self.encoder_layer, num_layers=n_en, norm=self.encoder_norm)

        self.decoder_norm = nn.LayerNorm(d_embed)
        self.decoder_layer = ts.TransformerDecoderLayer(d_model=d_embed, nhead=n_head, dim_feedforward=d_ff,
                                                        dropout=dropout, activation='relu')
        self.decoder = ts.TransformerDecoder(self.decoder_layer, num_layers=n_de, norm=self.decoder_norm)

        self.fc = nn.Linear(d_embed, vocab_size, bias=False)

    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask,
                memory_key_padding_mask):
        src = self.embed(src)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        tgt = self.embed(tgt)
        tgt = tgt.permute(1, 0, 2)
        tgt = self.pos_encoder(tgt)
        
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = output.permute(1, 0, 2)

        output = self.fc(output)
        return output
