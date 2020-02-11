#from Steel.utils import *
import os
os.chdir(os.path.dirname(__file__))
import sys
#sys.path.append('/home/chanhu/桌面/Kaggle/Steel/')
sys.path.append("/home/chanhu/桌面/Kaggle/")
import time
from Steel.cls_seg.dataset import *
from Steel.cls_seg.model.model_unet import *
from Steel.cls_seg.model.model_fpn import *
import pandas as pd
from Steel.utils import Meter, epoch_log
#from Steel.cls_seg.utils import softdiceloss, FocalLoss, Meter_Softmax# epoch_log
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from Steel.cls_seg.kaggle import criterion

#warnings.filters('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class config:
    csv = {'train': '/home/chanhu/桌面/Kaggle/Steel/cls_seg/train.csv',
           'test': '../input/severstal-steel-defect-detection/sample_submission.csv'}
    images = {'train': '/home/chanhu/桌面/Kaggle/Steel/inputs/train_images/',
              'test': '../input/severstal-steel-defect-detection/test_images/'}
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = softdiceloss()
    focal_loss = FocalLoss()
    n_epochs = 100
    class_weight = [5, 5, 2, 5]
    min_size     = [600, 600, 1000, 2000]

def adjust_learning_rate(optimizer, init_lr=1e-2, gamma=0.1):
    lr = init_lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def val_model(model, val_loader):

    meter = Meter(phase="val", epoch=1)
    running_loss = 0.0
    model.eval()
    for inputs, masks, labels in val_loader:
        inputs, masks, labels = inputs.float().cuda(), masks.long().cuda(), labels.float().cuda()
        logits = model(inputs)
        loss  = config.bce_loss(logits, masks)
        loss = 128*loss
        meter.update(masks, logits)
        running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    dices, iou = meter.get_metrics()
    torch.cuda.empty_cache()

    return val_loss, dices #, dice_neg, dice_pos

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

def train_model(train_csv, fold_index=0, checkPoint_start=0, lr=1e-2, batch_size=4, accumulate=True):

    model = PANet(num_class=4)

    if torch.cuda.is_available():
        model.cuda()
    i = 0
    iter = 0
    num_iters = 1500 * 1000 * (8 // batch_size)
    iter_valid = 1500 * (8 // batch_size)
    iter_save = [0, num_iters-1]\
                + list(range(0, num_iters, 1500 * (8 // batch_size)))
    start_iter = 0
    epoch = 0
    accumulate_step = 0
    start_epoch = 0

    if accumulate:
        accumulate_step = 32 // batch_size

    out_dir = os.path.dirname(__file__)  #
    checkPoint = os.path.join(out_dir, 'checkpoint')  #
    os.makedirs(checkPoint, exist_ok=True)  #

    # Define Kfold
    folds = pd.read_csv(train_csv)
    train_df = folds[folds['fold'] != fold_index]
    valid_df = folds[folds['fold'] == fold_index]
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    print(len(train_df), len(valid_df))

    train_dataset = SteelDataset2(
        mode='train',
        image_df=train_df,
        image_dir=config.images['train'],
        classes=[1, 2, 3, 4],
        if_softmax=True,
        augment=augment_train
    )

    val_dataset = SteelDataset2(
        mode='val',
        image_df=valid_df,
        image_dir=config.images['train'],
        classes=[1, 2, 3, 4],
        if_softmax=True,
        augment=augment_test
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=FourBalanceClassSampler(train_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=0.0001)

    if checkPoint_start != 0:
        initial_checkpoint = os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start))
        model.load_state_dict(torch.load(initial_checkpoint))
        initial_optimizer = initial_checkpoint.replace('_model.pth', '_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint = torch.load(initial_optimizer)
            start_iter = checkpoint['iter']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    iter = start_iter

    while iter < num_iters:
        #running_loss = 0.0
        optimizer.zero_grad()
        for t, (inputs, masks, labels) in enumerate(train_loader):
            batch_size = inputs.size(0)
            iter = i + start_iter

            epoch = (iter - start_iter) * batch_size / len(train_dataset) + start_epoch

            if (iter % iter_valid == 0) and (iter != checkPoint_start):
                ###################
                # val process
                ###################
                val_loss, iou, dice, dice_neg, dice_pos = val_model(model, val_loader)
                print( "iter: %d | Val Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice neg: %0.4f, | dice pos: %0.4f" % (iter, val_loss, iou, dice, dice_neg, dice_pos))

                #running_loss = 0.0

            if iter > 1500 * (8 // batch_size) * 60:
                adjust_learning_rate(optimizer, init_lr=lr, gamma=0.1)
            elif iter > 1500 * (8 // batch_size) * 100:
                adjust_learning_rate(optimizer, init_lr=lr, gamma=0.1**2)

            #poly_lr_scheduler(optimizer, init_lr=lr, max_iter=num_iters, iter=iter, )
            #if (iter % iter_valid == 0): print("lr:", optimizer.param_groups[0]['lr'])

            if iter in iter_save:
                torch.save(model.state_dict(), checkPoint + '/%08d_model.pth' % (iter))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'epoch': epoch}, checkPoint + '/%08d_optimizer.pth' % (iter))
                pass

            #######################
            # Training Process
            #######################

            model.train()
            inputs, masks, labels = inputs.float().cuda(), masks.long().cuda(), labels.float().cuda()
            logits = model(inputs)
            loss = config.bce_loss(logits, masks)
            loss = 128 * loss
            loss.backward()
            #weight=config.class_weight)
            (loss / accumulate_step).backward()
            if (iter % accumulate_step) == 0:
                optimizer.step()
                optimizer.zero_grad()

            i += 1
            if iter == num_iters:
                break

if __name__ == "__main__":
    seed_everything(1234)
    train_model(train_csv=config.csv['train'], fold_index=0, checkPoint_start=114000, lr=4e-4, batch_size=8, accumulate=True)
