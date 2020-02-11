import torch
import torch.nn.functional as F
import numpy as np

def one_hot_encode_truth(truth, num_class=4):
    one_hot = truth.repeat(1,num_class,1,1)
    arange  = torch.arange(1,num_class+1).view(1,num_class,1,1).to(truth.device)
    one_hot = (one_hot == arange).float()
    return one_hot


def one_hot_encode_predict(predict, num_class=4):
    value, index = torch.max(predict, 1, keepdim=True)

    value  = value.repeat(1,num_class,1,1)
    index  = index.repeat(1,num_class,1,1)
    arange = torch.arange(1,num_class+1).view(1,num_class,1,1).to(predict.device)

    one_hot = (index == arange).float()
    value = value*one_hot
    return value

#def criterion(logit, truth, weight=[5,5,2,5]):
def criterion(logit, truth, weight=None):
    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
    #print(logit.size())
    truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)
    #print(truth.size())
    if weight is not None:
        weight = torch.FloatTensor([1]+weight).cuda()
    loss = F.cross_entropy(logit, truth, weight=weight, reduction='none')

    loss = loss.mean()
    return loss


if __name__ == '__main__':
    pass
    #if 9000 % 3000:
        #print(1)
    #logit = torch.ones((2, 5, 256, 1600))
    #truth = np.zeros((256, 1600, 4))
    #truth = truth * [1, 2, 3, 4]
    #truth = truth.reshape(-1, 4)
    #print(truth)
    #truth = truth.max(-1).reshape(256, 1600, 1)
    #print(truth)
    #print(truth)
