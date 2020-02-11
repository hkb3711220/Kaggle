import os
os.chdir(os.path.dirname(__file__))
import sys
sys.path.append("/home/chanhu/桌面/Kaggle/")
from Steel.cls_seg.dataset import *
from Steel.cls_seg.model.model_unet import *
from Steel.cls_seg.utils import post_process
from Steel.cls_seg.model.model_fpn import *
from Steel.cls_seg.train_model2 import config
from Steel.cls_seg.kaggle import *
import pandas as pd
import torch.nn as nn
import warnings
import torch
warnings.filterwarnings('ignore')
from tqdm import tqdm

def sharpen(p, t=0.5):
    if t != 0:
        return p ** t
    else:
        return p

def dice_single_channel(probability, truth, eps = 1E-9):
    p = probability.reshape(-1).astype(float)
    t = truth.reshape(-1).astype(float)
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice

def dice_channel(probability, truth):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    #with torch.no_grad():
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :])
            mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel

def remove_small_one(predict, min_size):
    H, W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H, W), np.bool)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = True
    return predict

def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b,c] = remove_small_one(predict[b,c], min_size[c])
    return predict





def evaluate_model_softmax(net, test_dataset, threshold=0.5, augment=[]):
    # def sharpen(p,t=0):
    def sharpen(p, t=0):
        if t != 0:
            return p ** t
        else:
            return p

    dice_scores = []
    threshold_pixel = [threshold] * 4
    for input, masks, _ in tqdm(test_dataset):
        input, masks = input.to(device), masks.float().to(device) #in enumerate(tqdm(test_dataset)):

        batch_size, C, H, W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1:  # null
                logit = net(input)  # net(input)
                probability = torch.softmax(logit, 1)

                probability_mask = sharpen(probability, 0)
                num_augment += 1

            if 'flip_lr' in augment:
                logit = net(torch.flip(input, dims=[3]))
                probability = torch.softmax(torch.flip(logit, dims=[3]), 1)

                probability_mask += sharpen(probability)
                num_augment += 1

            if 'flip_ud' in augment:
                logit = net(torch.flip(input, dims=[2]))
                probability = torch.softmax(torch.flip(logit, dims=[2]), 1)

                probability_mask += sharpen(probability)
                num_augment += 1

            probability_mask = probability_mask / num_augment
            probability = probability_mask.clone()

            probability_mask = one_hot_encode_predict(probability_mask)
            probability_mask = probability_mask.detach().cpu().numpy()
            predict_mask = (probability_mask > np.array(threshold_pixel).reshape(1, 4, 1, 1)).astype(int)
            predict_mask = remove_small(predict_mask, min_size=config.min_size)
            truth_mask = one_hot_encode_truth(masks)
            truth_mask = truth_mask.detach().cpu().numpy()

            dice_score = dice_channel(predict_mask, truth_mask)
            dice_scores.append(dice_score)


            #for fname, preds in zip(fnames, predict_mask):
                #for cls, pred in enumerate(preds):
                    #rle = mask2rle(pred)
                    #name = fname + f"_{cls + 1}"
                    #predictions.append([name, rle])

    #df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])

    return np.mean(dice_scores)

def evalute_model(test_loader, model, threshold):

    dice_scores = []

    model.eval()
    for inputs, masks, _ in tqdm(test_loader):
        images, masks = inputs.to(device), masks.float().to(device)
        predicts = 0.0
        with torch.set_grad_enabled(False):

            outputs2 = model(torch.flip(images, dims=[2]))
            outputs3 = model(torch.flip(images, dims=[3]))

            predicts += sharpen(torch.sigmoid(model(images)).detach().cpu().numpy())
            predicts += sharpen(torch.sigmoid(torch.flip(outputs2, dims=[2])).detach().cpu().numpy())
            predicts += sharpen(torch.sigmoid(torch.flip(outputs3, dims=[3])).detach().cpu().numpy())

            predicts /= 3

            pred_all = np.zeros_like(predicts)
            for i, preds in enumerate(predicts):
                for cls, pred in enumerate(preds):
                    pred, _ = post_process(pred, threshold, min_size=config.min_size[cls])
                    pred_all[i, cls, :, :] = pred

            dice_score = dice_channel(pred_all, masks.detach().cpu().numpy())
            dice_scores.append(dice_score)

    return np.mean(dice_scores)

def optimizer_threshold(val_predicts, val_truths, cls):

    thresholds = list(np.arange(0.1, 1.0, 0.1))
    val_dice_coef = [dice_single_channel(val_predicts, val_truths, th) for th in tqdm(thresholds)]
    threshold_best_index = np.argmax(val_dice_coef)
    dice_coef_best = val_dice_coef[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    print(threshold_best)

    plt.plot(thresholds, val_dice_coef)
    plt.plot(threshold_best, dice_coef_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("dice_coef")
    plt.title("Threshold vs dice coef ({}, {})".format(threshold_best, dice_coef_best))
    plt.legend()




if __name__ == "__main__":

    #min_size = [600, 600, 1000, 2000]

    device = torch.device('cuda:0')
    folds = pd.read_csv(config.csv['train'])
    valid_df = folds[folds['fold'] == 0]
    valid_df.reset_index(drop=True, inplace=True)

    val_dataset = SteelDataset2(
        mode='val',
        image_df=valid_df,
        image_dir=config.images['train'],
        classes=[1, 2, 3, 4],
        augment=augment_test,
        if_softmax=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    seg_model = ResNet_unet(num_class=5)#PANet(num_class=5)
    seg_model.cuda()
    seg_model.load_state_dict(torch.load('/home/chanhu/桌面/Kaggle/Steel/cls_seg/checkpoint/00114000_model.pth'))

    thresholds = list(np.arange(0.1, 1.0, 0.1))
    best_scores = 0.0
    best_thresholds = 0.0
    for thres in thresholds:
        dice_scores_mean = evaluate_model_softmax(seg_model, val_loader, threshold=thres, augment=['null', 'flip_lr','flip_ud'])#val_loader, seg_model, threshold=thres)
        print('threshold', thres, 'dice scores', dice_scores_mean)
        if dice_scores_mean > best_scores:
            best_scores = dice_scores_mean
            best_thresholds = thres
    print(best_thresholds)
    #optimizer_threshold(val_predicts, val_truths, cls=1)

    #score = dice_channel(val_predicts, val_truths, threshold=0.5)
    #opt = Optimizedthreshold()
    #opt.fit(val_predicts, val_truths)
    #coefficients = opt.coefficients()
    #print(coefficients)






