import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Steel.cls_seg.kaggle import *
from torch.autograd import Variable
import cv2

def dice_single_channel(probability, truth, threshold, eps = 1e-12):
    p = (probability.view(-1) > threshold).float()
    t = truth.view(-1).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice

def dice_channel_torch(probability, truth, threshold):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel

def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X.detach().cpu().numpy())
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5):

    with torch.no_grad():

        #probability = torch.softmax(probability, 1)
        #probability = one_hot_encode_predict(probability)
        #truth = one_hot_encode_truth(truth)

        #batch_size, num_class, H, W = truth.shape
        dice = dice_channel_torch(probability, truth, threshold=threshold)

        #probability = probability.view(batch_size, num_class, -1)
        #truth = truth.view(batch_size, num_class, -1)
        #p = (probability > threshold).float()
        #t = (truth > 0.5).float()

        #for i in range(batch_size):
            #for j in range(num_class):
                #channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                #mean_dice_channel += channel_dice/(batch_size * channel_num)


        #t_sum = t.sum(-1)
        #p_sum = p.sum(-1)

        #d_neg = (p_sum < sum_threshold).float()
        #d_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-12)

        #neg_index = (t_sum == 0).float()
        #pos_index = 1 - neg_index

        #num_neg = neg_index.sum()
        #num_pos = pos_index.sum(0)
        #dn = (neg_index * d_neg).sum() / (num_neg + 1e-12)
        #dp = (pos_index * d_pos).sum(0) / (num_pos + 1e-12)

        # ----

        #dn = dn.item()
        #dp = list(dp.data.cpu().numpy())
        #num_neg = num_neg.item()
        #num_pos = list(num_pos.data.cpu().numpy())

    return dice #dn, dp, num_neg, num_pos

class Meter_Softmax:
    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self, phase, epoch, threshold=0.4):
        self.base_threshold = threshold  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.softmax(outputs,1)
        probability = one_hot_encode_predict(probs)
        truth_mask  = one_hot_encode_truth(targets)
        dice = dice_channel_torch(probability, truth_mask, threshold=self.base_threshold)#metric(probability, truth_mask, self.base_threshold) #, dice_neg, dice_pos, _,
        #print(dice)# _
        self.base_dice_scores.append(dice.item())
        #self.dice_pos_scores.append(dice_pos)
        #self.dice_neg_scores.append(dice_neg)
        #preds = predict(probs, self.base_threshold)
        #iou = compute_iou_batch(preds, targets, classes=[1])
        #self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        #dice_neg = np.mean(self.dice_neg_scores)
        #dice_pos = np.mean(self.dice_pos_scores)
        #dices = [dice, dice_neg, dice_pos]
        #iou = np.nanmean(self.iou_scores)

        return dice#, iou


def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (
    epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels.detach().cpu().numpy())  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)

    return iou


class softdiceloss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(softdiceloss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num

        return score


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=1, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs  = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE     = F.binary_cross_entropy(inputs, targets, reduction='mean') #log(p)
        BCE_EXP = torch.exp(-BCE) #p
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=100, power=0.9):
   """Polynomial decay of learning rate
       :param init_lr is base learning rate
       :param iter is a current iteration
       :param lr_decay_iter how frequently decay occurs, default is 1
       :param max_iter is number of maximum iterations
       :param power is a polymomial power
   """
   if iter % lr_decay_iter or iter > max_iter:
       return optimizer
   lr = init_lr*(1 - iter/max_iter)**power
   for param_group in optimizer.param_groups:
       param_group['lr'] = lr
   #return lr

class FocalLoss_softmax(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()