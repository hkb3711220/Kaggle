import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor
from pytorch_pretrained_bert import BertModel
import torch

class score(torch.nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(score, self).__init__()
        self.score = torch.nn.Sequential(
                     torch.nn.Linear(embed_dim, hidden_dim),
                     torch.nn.ReLU(inplace=True),
                     torch.nn.Dropout(0.2),
                     torch.nn.Linear(hidden_dim, hidden_dim),
                     torch.nn.ReLU(inplace=True),
                     torch.nn.Dropout(0.2))

    def forward(self, x):
        return self.score(x)

class mention_score(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(mention_score, self).__init__()
        self.score = torch.nn.Sequential(
                     score(input_dim, hidden_dim),
                     torch.nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, x):

        output = [self.score(_x) for _x in x]

        return output

class mentionpair_score(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(mentionpair_score, self).__init__()
        self.score = torch.nn.Sequential(
                     score(input_dim, hidden_dim),
                     torch.nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, g1, g2, g1_score, g2_score):

        element_wise = g1 * g2
        pair_score   = self.score(torch.cat((g1, g2, element_wise), dim=-1))

        return torch.sum(torch.cat((g1_score, g2_score, pair_score), dim=-1), dim=1, keepdim=True)

class new_model(torch.nn.Module):

    def __init__(self):
        super(new_model, self).__init__()
        self.bert           = BertModel.from_pretrained('bert-base-uncased')
        self.span_extractor = EndpointSpanExtractor(768, "x,y,x*y")
        self.mention_score  = mention_score(2304, 150) #input = (batch_size,2304), output=(batch_size,1)
        self.pair_score     = mentionpair_score(2304*3, 150) #input = (batch_size,2304), output=(batch_size,1)
        self.softmax        = torch.nn.Softmax(dim=1)

    def forward(self, sent, offsets):

        bert_output, _  = self.bert(sent, output_all_encoded_layers=False)
        span_repres     = self.span_extractor(bert_output, offsets)
        span_repres     = torch.unbind(span_repres, dim=1)
        scores          = self.mention_score(span_repres)

        ap_score = self.pair_score(span_repres[2], span_repres[0], scores[2], scores[0])
        bp_score = self.pair_score(span_repres[2], span_repres[1], scores[2], scores[1])
        nan_score = torch.zeros_like(ap_score)

        output = torch.cat((ap_score, bp_score, nan_score), dim=1)
        output = self.softmax(output)

        return output
