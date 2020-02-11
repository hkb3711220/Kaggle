from torch.utils.data import *
import torch
import numpy as np
import pandas as pd
import cv2
import random
import albumentations
import matplotlib.pyplot as plt
from torchvision import transforms

def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort=pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    df = df.drop('sort', axis=1)
    return df

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T

class SteelDataset2(torch.utils.data.Dataset):

    def __init__(self, image_df, image_dir, classes, mode,
                 label_smoothing=True,
                 if_softmax=False, augment=None):
        self.image_df = image_df
        self.image_dir = image_dir
        self.mode = mode
        self.augment = augment
        self.classes = classes
        self.num_class = len(classes)
        self.label_smoothing = label_smoothing
        self.if_softmax = if_softmax
        self.preprocess()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]
        #print(image_id)
        image = cv2.imread(self.image_dir + image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            h, w = image.shape[0], image.shape[1]
            mask = np.zeros((h, w, self.num_class))
            label = np.expand_dims(self.image_labels[idx], -1)

            rle = [
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_1', 'EncodedPixels'].values[0],
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_2', 'EncodedPixels'].values[0],
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_3', 'EncodedPixels'].values[0],
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_4', 'EncodedPixels'].values[0]]

            for i, r in enumerate(rle): mask[:, :, i] = rle2mask(r, image.shape)

            if self.augment:
                augmented = self.augment(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            if self.if_softmax:
                H, W, num_class = mask.shape
                mask = mask * self.classes
                mask = mask.reshape(-1, 4)
                mask = mask.max(-1).reshape(H, W, 1)

            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)

            return image, mask, label

        elif self.mode == 'val':
            h, w = image.shape[0], image.shape[1]
            mask = np.zeros((h, w, self.num_class))
            label = np.expand_dims(self.image_labels[idx], -1)

            rle = [
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_1', 'EncodedPixels'].values[0],
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_2', 'EncodedPixels'].values[0],
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_3', 'EncodedPixels'].values[0],
                self.image_df.loc[self.image_df['ImageId_ClassId'] == image_id + '_4', 'EncodedPixels'].values[0]]

            for i, r in enumerate(rle): mask[:, :, i] = rle2mask(r, image.shape)

            if self.augment:
                augmented = self.augment(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                #if np.sum(aug_mask) == 0 and np.sum(mask) != 0:
                    #image = image
                    #mask  = mask
                #else:
                    #image = aug_image
                    #mask  = aug_mask

            if self.if_softmax:
                H, W, num_class = mask.shape
                mask = mask * self.classes
                mask = mask.reshape(-1, 4)
                mask = mask.max(-1).reshape(H, W, 1)

            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)

            return image, mask, label

        elif self.mode == 'test':

            if self.augment:
                augmented = self.augment(image=image)
                image = augmented['image']

            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)

            return image

    def preprocess(self):

        self.image_df["ImageId"] = self.image_df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
        self.image_df.fillna('', inplace=True)
        self.image_df['Label'] = (self.image_df['EncodedPixels'] != '').astype(np.int32)
        #print(self.image_df['Label'].head())

        sub_df = self.image_df[["ImageId", "Label"]].groupby("ImageId").sum()
        imageid_cls_df = pd.DataFrame()
        imageid_cls_df["ImageId"] = self.image_df['ImageId'].unique()
        imageid_cls_df["Labels"] = (sub_df.values > 0).astype(np.int32)
        #for clss in self.classes:
            #x = []
            #for img_name in imageid_cls_df["ImageId"].values:
                #imgid_classid = img_name+"_{}".format(clss)
                #img_info = self.image_df[self.image_df["ImageId_ClassId"] == imgid_classid]
                #print(img_info['Label'])
                #if img_info['Label'].values != 0:
                    #x.append(1)
                #else:
                    #x.append(0)
            #imageid_cls_df["class_{}".format(clss)] = x
        #print(imageid_cls_df.head())

        self.image_list    = imageid_cls_df["ImageId"]
        self.image_labels  = imageid_cls_df["Labels"]
        #self.image_classes = imageid_cls_df[['class_1', 'class_2', 'class_3', 'class_4']].values
        self.image_df      = df_loc_by_list(self.image_df, 'ImageId_ClassId', [u + '_%d' % c for u in self.image_list for c in [1, 2, 3, 4]])

class FourBalanceClassSampler(Sampler):

    def __init__(self, input_dataset):
        self.dataset = input_dataset

        label = (self.dataset.image_df['Label'].values)
        label = label.reshape(-1, 4)
        label = np.hstack([label.sum(1, keepdims=True) == 0, label]).T

        self.neg_index = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        # assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4 * num_neg

    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg, pos1, pos2, pos3, pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


augment_train = albumentations.Compose([
    #albumentations.RandomCrop(height=224, width=1600),
    albumentations.CropNonEmptyMaskIfExists(height=224, width=1600),
    albumentations.Resize(256, 1600),
    #albumentations.OneOf([
        #albumentations.RandomGamma(gamma_limit=(60, 120), p=0.5),
        #albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
    #]),
    albumentations.OneOf([
     albumentations.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
     albumentations.GridDistortion(p=0.5),
     albumentations.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)
    ], p=0.5),
    albumentations.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=20, border_mode=0, p=1),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1)
], p=1)

#augment_train = albumentations.Compose([
        #albumentations.Resize(256, 1600),
        #albumentations.OneOf([
            #albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
            #albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            #albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        #]),
        #albumentations.OneOf([
            #albumentations.Blur(blur_limit=4, p=1),
            #albumentations.MotionBlur(blur_limit=4, p=1),
            #albumentations.MedianBlur(blur_limit=4, p=1)
        #], p=0.5),
        #albumentations.HorizontalFlip(p=0.5),
        #albumentations.VerticalFlip(p=0.5),
        #albumentations.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        #albumentations.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),

        #albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=0.2,
                                        #interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
        #albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    #], p=1)

augment_test = albumentations.Compose([
        albumentations.Resize(256, 1600),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ], p=1)


def run_check_train_dataset(augment=None):
    dataset = SteelDataset2(
        mode='train',
        image_df=pd.read_csv(config.csv['train']),
        image_dir=config.images['train'],
        classes=[1, 2, 3, 4],
        augment=augment
    )
    i = 0

    image, mask, label = dataset[i]
    print(label)
    img_ = transforms.ToPILImage()(image)
    mask_ = mask.cpu().numpy()
    mask_ = mask_.transpose((1, 2, 0))
    plt.figure(figsize=(40, 10))
    plt.subplot(5, 1, 1)
    plt.imshow(img_)
    plt.axis('off')
    for i in range(len([1, 2, 3, 4])):
        plt.subplot(5, 1, 2 + i)
        plt.imshow(mask_[:, :, i], cmap='gray')
        plt.axis('off')
    plt.show()

def run_check_data_loader():

    dataset = SteelDataset2(
        mode      = 'train',
        image_df  = pd.read_csv(config.csv['train']),
        image_dir = config.images['train'],
        classes   = [1,2,3,4],
        augment   = None
    )
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = FourBalanceClassSampler(dataset),
        batch_size  = 8,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        #collate_fn  = null_collate
    )

    for t,(input, truth, labels) in enumerate(loader):
        print('----t=%d---'%t)
        print('')
        print('input', input.shape)
        print('truth', truth.shape)
        print('')
        print(labels)


if __name__ == '__main__':
    pass
    class config:
        pass
    #import os
    #os.chdir(os.path.dirname(__file__))
    #class config:
       #csv = {'train': './train.csv',
               #'test': '../input/severstal-steel-defect-detection/sample_submission.csv'}
       #images = {'train': '/home/chanhu/桌面/Kaggle/Steel/inputs/train_images/',
                  #'test': '../input/severstal-steel-defect-detection/test_images/'}
    #run_check_train_dataset()
    #run_check_data_loader()