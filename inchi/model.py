import math
import numpy as np
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import timm
import config
from tnt import *
from fairseq import utils
from fairseq.models import *
from fairseq.modules import *

class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerDecoder(FairseqIncrementalDecoder):

    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})

        layer_parameter = Namespace({
            'decoder_embed_dim': dim,
            'decoder_attention_heads': num_head,
            'attention_dropout': 0.1,
            'dropout': 0.1,
            'decoder_normalize_before': True,
            'decoder_ffn_embed_dim': ff_dim,
            })
        self.layer = nn.ModuleList(modules=[TransformerDecoderLayer(layer_parameter) for i in range(num_layer)])
        self.layer_norm = nn.LayerNorm(dim)


    def forward(self, x, mem, x_mask):
        for layer in self.layer:
            x = layer(x, mem, self_attn_mask=x_mask)[0]
        x = self.layer_norm(x)
        return x  # T x B x C

    def forward_one(self,
            x   : torch.Tensor,
            mem : torch.Tensor,
            incremental_state : Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    )-> torch.Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state)[0]
        x = self.layer_norm(x)
        return x

class Encoder(nn.Module):

    def __init__(self, model_name: str, pretrained: bool = False):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.cnn.classifier.in_features
        self.linear = nn.Linear(in_features, config.decoder_dim)

    def forward(self, x):
        features = self.cnn.forward_features(x)
        features = features.permute(0, 2, 3, 1)
        features = self.linear(features)
        return features


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
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


class Decoder(nn.Module):

    def __init__(self, vocab_size: int, num_dims: int, dim_feedforward:int,
                 max_len: int, num_headers: int = 8,
                 num_layer: int = 6):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_dims = num_dims
        self.embed_layer = nn.Embedding(vocab_size, num_dims)
        self.pos_embed_layer = PositionalEmbedding(d_model=num_dims, max_len=max_len)
        self.decoder = TransformerDecoder(dim=num_dims, ff_dim=dim_feedforward, num_head=num_headers,
                                          num_layer=num_layer)
        #for slow predict
        #decoder_layer = nn.TransformerDecoderLayer(d_model=num_dims, dim_feedforward=dim_feedforward, nhead=num_headers)
        #self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layer)
        self.linear = nn.Linear(num_dims, vocab_size)

        self.embed_layer.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, labels, caption_lengths):

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        sorted_encoder_out = encoder_out[sort_ind].permute(1, 0, 2)
        sorted_labels = labels[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        text_mask = np.triu(np.ones((max(decode_lengths) + 1, max(decode_lengths) + 1)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask) == 1).to(device)

        # !TODO: memory mask
        # memory_mask = np.triu(np.ones((max(decode_lengths) + 1, sorted_encoder_out.size(0))), k=1).astype(np.uint8)
        # memory_mask = torch.autograd.Variable(torch.from_numpy(memory_mask) == 1).to(device)

        embed_tgt = self.embed_layer(sorted_labels).permute(1, 0, 2)  # (max_caption_length, bs, num_dims)
        embed_tgt = self.pos_embed_layer(embed_tgt)  # (max_caption_length, bs, num_dims)
        decoder_out = self.decoder(embed_tgt, sorted_encoder_out, text_mask) #(max_caption_length, bs, num_dims)
        decoder_out = decoder_out.permute(1, 0, 2)
        # based on pytorch transformer (slow predict)
        #decoder_out = self.decoder(tgt=embed_tgt, memory=sorted_encoder_out, tgt_mask=text_mask) # (
        decoder_out = self.linear(decoder_out) # (bs, max_caption_length, vocab_size)

        return decoder_out, sorted_labels, decode_lengths

    def slow_predict(self, encoder_out):

        eos = config.STOI['<eos>']
        pad = config.STOI['<pad>']

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.permute(1, 0, 2)

        result = torch.full((batch_size, self.max_len), pad, dtype=torch.long, device=device)
        result[:, 0] = config.STOI['<sos>']
        for t in range(1, self.max_len):
            tgt_emb = self.embed_layer(result[:, :t]).transpose(0, 1)
            tgt_emb = self.pos_embed_layer(tgt_emb)
            decode_out = self.decoder(tgt=tgt_emb, memory=encoder_out)
            prob_output_t = self.linear(decode_out)[-1, :, :]
            output_t = prob_output_t.data.topk(1)[1].squeeze()
            result[:, t] = output_t
            if ((output_t == eos) | (output_t == pad)).all():  break

        predict = result[:, 1:]

        return predict

    def predict(self, encoder_out):
        """
        https://www.kaggle.com/c/bms-molecular-translation/discussion/231190
        """

        eos = config.STOI['<eos>']
        pad = config.STOI['<pad>']

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.permute(1, 0, 2)

        result = torch.full((batch_size, self.max_len), pad, dtype=torch.long, device=device)
        text_pos = self.pos_embed_layer.pe
        result[:, 0] = config.STOI['<sos>']

        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[torch.Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
        )
        for t in range(self.max_len - 1):
            last_token = result[:, t]  # (batch_size)
            tgt_emb = self.embed_layer(last_token)  # (batch_size, num_dims)
            tgt_emb = tgt_emb + text_pos[t, :]
            tgt_emb = tgt_emb.reshape(1, batch_size, self.num_dims)
            decode_out = self.decoder.forward_one(tgt_emb, encoder_out, incremental_state=incremental_state)
            decode_out = decode_out.reshape(batch_size, self.num_dims)
            prob_output_t = self.linear(decode_out)
            k = torch.argmax(prob_output_t, -1)
            result[:, t + 1] = k
            if ((k == eos) | (k == pad)).all():
                break

        predict = result[:, 1:]

        return predict

    def infer(self, encoder_out):

        eos = config.STOI['<eos>']
        pad = config.STOI['<pad>']

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.permute(1, 0, 2)

        result = torch.full((batch_size, self.max_len), pad, dtype=torch.long, device=device)
        result_matrix = torch.zeros((batch_size, self.max_len, self.vocab_size), device=device)
        result_matrix[:, :, pad] = 1
        text_pos = self.pos_embed_layer.pe
        result[:, 0] = config.STOI['<sos>']
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[torch.Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
        )

        for t in range(self.max_len - 1):
            last_token = result[:, t]  # (batch_size)
            tgt_emb = self.embed_layer(last_token)  # (batch_size, num_dims)
            tgt_emb = tgt_emb + text_pos[t, :]
            tgt_emb = tgt_emb.reshape(1, batch_size, self.num_dims)
            decode_out = self.decoder.forward_one(tgt_emb, encoder_out, incremental_state=incremental_state)
            decode_out = decode_out.reshape(batch_size, self.num_dims)
            prob_output_t = self.linear(decode_out)
            k = torch.argmax(prob_output_t, -1)
            result_matrix[:, t + 1, :] = prob_output_t
            result[:, t + 1] = k
            if ((k == eos) | (k == pad)).all():
                break

        predict_matrix = result_matrix[:, 1:]

        return predict_matrix


class InchiModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_model = Encoder(model_name=config.encoder_model_name,
                                     pretrained=config.encoder_model_pretrained)
        self.decoder_model = Decoder(vocab_size=vocab_size,
                                     num_dims=config.decoder_dim,
                                     dim_feedforward=config.dim_feedforward,
                                     max_len=config.max_len,
                                     num_headers=config.num_head,
                                     num_layer=config.num_layer)

    def forward(self, images, labels, decode_lengths):
        encoder_output = self.encoder_model(images)
        train_output, sorted_labels, sorted_decode_lengths = self.decoder_model(encoder_output,
                                                                                labels=labels,
                                                                                caption_lengths=decode_lengths)

        return train_output, sorted_labels, sorted_decode_lengths

    def predict(self, images):
        encoder_output = self.encoder_model(images)
        predictions = self.decoder_model.predict(encoder_output)

        return predictions

    def infer(self, images):
        encoder_output = self.encoder_model(images)
        predictions = self.decoder_model.infer(encoder_output)

        return predictions
