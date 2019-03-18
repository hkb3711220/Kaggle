from torch import nn
import torch

class DistanceEmbed(nn.Module):

    def __init__(self, dimension=20):

        super(DistanceEmbed, self).__init__()

        self.buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]
        self.Embedding = nn.Embedding(len(self.buckets)+1, dimension)

    def forward(self, x):
        return self.Embedding(self._bs(x))

    def _bs(self, lengths):

        return torch.tensor([sum([True for i in self.buckets if num >= i]) for num in lengths], requires_grad=False)

class FFNN(nn.Module):

    def __init__(self, embeds_dim, hidden_dim=150):
        super(FFNN, self).__init__()
        self.FFNN1 = nn.Sequential(
                     nn.Linear(embeds_dim, hidden_dim),
                     nn.ReLU(),
                     nn.Dropout(0.20),
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.ReLU(),
                     nn.Dropout(0.20))

    def forward(self, x):
        return self.FFNN1(x)

class Attention(nn.Module):

    def __init__(self, features_dim, step_dim, bias=True, **kwargs):

        """
        https://www.kaggle.com/jannen/reaching-0-7-fork-from-bilstm-attention-kfold

        """

        super(Attention, self).__init__(**kwargs)

        self.features_dim = features_dim

        self.bias = bias
        weight = torch.zeros(features_dim, 1)
        self.step_dim = step_dim
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        #print(self.weight.shape)

        if self.bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x):

        at = torch.mm(x.contiguous().view(-1, self.features_dim), self.weight).view(-1, self.step_dim)

        if self.bias: at = at + self.b

        at = torch.tanh(at)
        at = torch.exp(at)

        at = at / torch.sum(at, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(at, -1)

        return torch.sum(weighted_input, 1)
