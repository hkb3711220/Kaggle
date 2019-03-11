
import torch
from utils import *
from torch import nn


class model(nn.Module):

    def __init__(self, max_len=50, max_features=80000, embed_size=300, hidden=200, dim=150):

        super(model, self).__init__()
        self.embed1 = nn.Embedding(max_features, embed_size)
        self.dist_embed = DistanceEmbed()
        self.lstm  = nn.LSTM(embed_size, hidden, bidirectional=True, batch_first=True)
        self.FFNN1  = FFNN(hidden*2, dim)
        self.Attention = Attention(dim, max_len)
        self.score1 = nn.Sequential(FFNN(970, dim), nn.Linear(dim, 1))
        self.score2 = nn.Sequential(FFNN(970*2+1+20, dim), nn.Linear(dim, 1))
        self.pairwise = nn.PairwiseDistance(keepdim=True)
        self.soft_max = nn.Softmax(1)

    def forward(self, x):

        M, A, B, M_pos, A_pos, B_pos, M_A_dist, M_B_dist = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]

        M_A_dist = self.dist_embed(M_A_dist)
        M_B_dist = self.dist_embed(M_B_dist)
        M_spans  = self.span_representation(M, M_pos)
        A_spans  = self.span_representation(A, A_pos)
        B_spans  = self.span_representation(B, B_pos)

        Sim1 = self.pairwise(M_spans, A_spans)
        Sim2 = self.pairwise(M_spans, B_spans)

        M_score = self.score1(M_spans)
        A_score = self.score1(A_spans)
        B_score = self.score1(B_spans)

        Sa = self.score2(torch.cat((M_spans, A_spans, Sim1, M_A_dist), 1))
        Sb = self.score2(torch.cat((M_spans, B_spans, Sim2, M_B_dist), 1))

        Core_scoreA = M_score + A_score + Sa
        Core_scoreB = M_score + B_score + Sb

        output = self.soft_max(torch.cat((Core_scoreA, Core_scoreB, M_score), 1))
        print(output.size())

        return output

    def span_representation(self, x, x_pos):

        sentence_embed = self.embed1(x)
        dist_embed     = self.dist_embed(x_pos)
        lstm_output, _ = self.lstm(sentence_embed)
        a              = self.FFNN1(lstm_output)
        xt             = self.Attention(a)
        gi             = torch.cat((lstm_output[:, 0, :], lstm_output[:, -1, :], xt, dist_embed), 1)

        return gi


#import numpy as np
x1 = torch.ones([300, 50, ], dtype=torch.int64)
x2 = torch.ones([300], dtype=torch.int64)
x3 = x2
x4 = x2
x = [x1, x1, x1, x2, x3, x4, x2, x2]
model()(x)
