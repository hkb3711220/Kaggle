# Just share our model.
# We had not completed Stage2. 
# Because I select the code which does not have melspectrogram transform.
# Our team spend a lot of time doing this compettion.
# So sorry to them.

import gc
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from psutil import cpu_count
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import librosa


N_JOBS = cpu_count()
os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
os.environ['OMP_NUM_THREADS'] = str(N_JOBS)
DataLoader = partial(DataLoader, num_workers=N_JOBS)

class config:
    SEED = 520
    PATH = '../input/freesound-audio-tagging-2019'
    preprocessed_dir = '../input/all-data'
    dataset_dir = '../input/freesound-audio-tagging-2019'
    dataset = {'train_curated': os.path.join(PATH, 'train_curated'),
                'train_noisy':os.path.join(PATH, 'train_noisy'),
                'test':os.path.join(PATH, 'test')}
    transforms_dict = {'train': transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor()]),
                        'test': transforms.Compose([
                        transforms.ToTensor()])}
        
    mels = {'train_curated': preprocessed_dir + '/train_curated.pickle'}
    csvs = {
    'train_curated': dataset_dir + '/train_curated.csv',
    'train_noisy': dataset_dir +'/train_noisy.csv',
    'sample_submission': dataset_dir + '/sample_submission.csv',
    }
    sampling_rate = 44100
    duration = 2 # sec
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration

def envelope(y, rate, threshold):
    
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
            
    return mask

def create_mel_inputs(df, dataset, check=False, num=False):
    
    output = []
    for i in tqdm(range(len(df))):
        
        path = os.path.join(config.dataset[dataset], df.loc[i, 'fname'])
        melspectrogram = extract_melspectrogram(path, display=check)
        
        if check:
            plt.figure()
            img = Image.fromarray(melspectrogram)
            img = np.asarray(img)
            plt.imshow(img)
        
        output.append(melspectrogram)
        
    if num:
        with open('{}_{}.pickle'.format(dataset, num), mode='wb') as f:
            pickle.dump(output, f)  
    else:
        with open('{}.pickle'.format(dataset), mode='wb') as f:
            pickle.dump(output, f) 
            
def spectrogram_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def extract_melspectrogram(path, display=False):
    wave, sr = librosa.load(path, sr=config.sampling_rate)
    if 0 < len(wave):  # workaround: 0 length causes error
        wave , _ = librosa.effects.trim(wave)
    if len(wave) < config.samples:
        padding = config.samples - len(wave)    # add padding at both ends
        offset = padding // 2
        wave = np.pad(wave, (offset, config.samples - len(wave) - offset), config.padmode)

    melspectrogram = librosa.feature.melspectrogram(wave, sr, n_fft=config.n_fft,
                                                    hop_length=config.hop_length,
                                                    n_mels=config.n_mels, fmin=config.fmin,
                                                    fmax=config.fmax)
    melspectrogram = librosa.power_to_db(melspectrogram)
    melspectrogram = melspectrogram.astype(np.float32)

    if display:
        librosa.display.specshow(melspectrogram, y_axis='mel', x_axis='time')

    return spectrogram_to_color(melspectrogram)
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Model1_FATTestDataset(Dataset):
    
    def __init__(self, df, transforms):
        super().__init__()
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        image = Image.fromarray(self.df.loc[idx, 'data'], mode='RGB')
        time_dim, base_dim = image.size
        image = self.transforms(image).div_(255)
        
        return image

class Model2_FATTestDataset(Dataset):
    def __init__(self, fnames, mels, transforms, tta=5):
        super().__init__()
        self.fnames = fnames
        self.mels = mels
        self.transforms = transforms
        self.tta = tta
        
    def __len__(self):
        return len(self.fnames) * self.tta
    
    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)
        
        image = Image.fromarray(self.mels[new_idx], mode='RGB')
        time_dim, base_dim = image.size
        if time_dim == base_dim:
            image1, image2 = image, image
            image1 = self.transforms(image1).div_(255)
            image2 = self.transforms(image2).div_(255)
        elif time_dim > base_dim and time_dim < 2* base_dim:
            image1 = image.crop([0, 0, base_dim, base_dim])
            crop = random.randint(0, time_dim - base_dim)
            image2 = image.crop([crop, 0, crop + base_dim, base_dim])
            image1 = self.transforms(image1).div_(255)
            image2 = self.transforms(image2).div_(255)
        else:
            crop1 = random.randint(0, int(time_dim / 2))
            image1 = image.crop([crop1, 0, crop1 + base_dim, base_dim])
            crop2 = random.randint(int(time_dim / 2), time_dim - base_dim)
            image2 = image.crop([crop2, 0, crop2 + base_dim, base_dim])
            image1 = self.transforms(image1).div_(255)
            image2 = self.transforms(image2).div_(255)

        fname = self.fnames[new_idx]
        
        return [image1, image2], fname


class Model3_FATTestDataset(Dataset):
    def __init__(self, fnames, mels, transforms, tta=5):
        super().__init__()
        self.fnames = fnames
        self.mels = mels
        self.transforms = transforms
        self.tta = tta
        
    def __len__(self):
        return len(self.fnames) * self.tta
    
    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)
        
        image = Image.fromarray(self.mels[new_idx], mode='RGB') 
        time_dim, base_dim = image.size
        if time_dim == base_dim:
            image1, image2, image3 = image, image, image
            image1 = self.transforms(image1).div_(255)
            image2 = self.transforms(image2).div_(255)
            image3 = self.transforms(image3).div_(255)
        elif time_dim > base_dim and time_dim < 3 * base_dim:
            image1 = image.crop([0, 0, base_dim, base_dim])
            crop1 = random.randint(0, time_dim - base_dim)
            image2 = image.crop([crop1, 0, crop1 + base_dim, base_dim])
            crop2 = random.randint(0, time_dim - base_dim)
            image3 = image.crop([crop2, 0, crop2 + base_dim, base_dim])
            image1 = self.transforms(image1).div_(255)
            image2 = self.transforms(image2).div_(255)
            image3 = self.transforms(image3).div_(255)
        else:
            crop1 = random.randint(0, int(time_dim/3))
            image1 = image.crop([crop1, 0, crop1 + base_dim, base_dim])
            crop2 = random.randint(int(time_dim/3), int(time_dim*(2/3)))
            image2 = image.crop([crop2, 0, crop2 + base_dim, base_dim])
            crop3 = random.randint(int(time_dim*(2/3)), time_dim - base_dim)
            image3 = image.crop([crop3, 0, crop3+ base_dim, base_dim])
            image1 = self.transforms(image1).div_(255)
            image2 = self.transforms(image2).div_(255)
            image3 = self.transforms(image3).div_(255)
        
        fname = self.fnames[new_idx]
        
        return [image1, image2, image3], fname
class Model4_FATTestDataset(Dataset):
    def __init__(self, fnames, mels, transforms, tta=5):
        super().__init__()
        self.fnames = fnames
        self.mels = mels
        self.transforms = transforms
        self.tta = tta
        
    def __len__(self):
        return len(self.fnames) * self.tta
    
    def __getitem__(self, idx):
        
        new_idx = idx % len(self.fnames)
        
        image = Image.fromarray(self.mels[new_idx], mode='RGB')
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)

        fname = self.fnames[new_idx]
        
        return image, fname

class ConvBlock1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock1, self).__init__()
        self.conv_max = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        return self.conv_max(x)
        
class ConvBlock(nn.Module):
    """
    use max pooling to downsampling
    """
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_max = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        return self.conv_max(x)

class Gate_ConvBlock(nn.Module):
    """
    
    <LARGE-SCALE WEAKLY SUPERVISED AUDIO CLASSIFICATION USING GATEDCONVOLUTIONAL NEURAL NETWORK>
    https://arxiv.org/pdf/1710.00343.pdf
    
    gated linear units (GLUs) as ac-tivation  to  replace  the  ReLU  [14]  activation  in  the  CRNNmodel.   
    The GLUs can control the amount of infor-mation of a T-F unit flow to the next layer.
    the GLUs can beregarded as an attention scheme.
    
    """
    

    def __init__(self, in_channels, out_channels):
        super(Gate_ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(2, stride=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        
        layer1_L      = self.conv1(x)
        layer1_L      = self.bn1(layer1_L)
        layer1_S      = torch.sigmoid(layer1_L)
        layer1_output = layer1_L * layer1_S
        layer2_L      = self.conv2(layer1_output)
        layer2_L      = self.bn2(layer2_L)
        layer2_S      = torch.sigmoid(layer2_L)
        layer2_output = layer2_L * layer2_S
        output        = self.pool(layer2_output)
        
        return output

class model_1(nn.Module):
    """
    basic idea:
    from last Convolution Block, we can get (N, 512, 4, 4)
    the last dim is time dim, second dim is melspectrogram. 
    ex. (N, 512, freq_dim, time_dim)
    Use Avg pooling to get the average value of freq_dim in every time.
    
    """

    def __init__(self):
        super(model_1, self).__init__()
        self.preprocess = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU())
        self.cnn = nn.Sequential(ConvBlock1(3, 32),#64
                                 ConvBlock1(32, 64),#32
                                 ConvBlock1(64, 128),#16 
                                 ConvBlock1(128, 256), #8
                                 ConvBlock1(256, 512), #4
                                 nn.AvgPool2d((4, 4))) 
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 80))

    def forward(self, x):
        
        x = self.preprocess(x)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output

class model_2_sub(nn.Module):

    def __init__(self):
        super(model_2_sub, self).__init__()
        self.preprocess = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU())
        self.cnn = nn.Sequential(ConvBlock(3, 32),
                                 ConvBlock(32, 64),
                                 ConvBlock(64, 128),
                                 ConvBlock(128, 256),
                                 ConvBlock(256, 512), 
                                 nn.AvgPool2d((4,4))) 
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 80))

    def forward(self, x):
        
        x = self.preprocess(x)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output

class model_2(nn.Module):
    
    def __init__(self):
        super(model_2, self).__init__()
        self.fea_extractor = model_2_sub()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image1, image2):
        
        fea1 = self.fea_extractor(image1)
        fea2 = self.fea_extractor(image2)
        fea1 = self.sigmoid(fea1)
        fea2 = self.sigmoid(fea2)
        fea1 = torch.unsqueeze(fea1, dim=1)
        fea2 = torch.unsqueeze(fea2, dim=1)
        feas = torch.cat((fea1, fea2), dim=1)
        feas = torch.mean(feas, dim=1) #GlobalAvgPool
        
        return feas
        
class model_3_sub(nn.Module):

    def __init__(self):
        super(model_3_sub, self).__init__()
        self.preprocess = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU())
        self.cnn = nn.Sequential(ConvBlock(3, 32),
                                 ConvBlock(32, 64),
                                 ConvBlock(64, 128),
                                 ConvBlock(128, 256),
                                 ConvBlock(256, 512),
                                 nn.AvgPool2d((4,4))) 

    def forward(self, x):
        
        x = self.preprocess(x)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)

        return x
        
class model_3(nn.Module):
    """
    basic idea:
    in this model, use multi input to represent different melspectrogram in different time.
    result:
    overfit training set (not so much)
    
    """
    
    
    def __init__(self):
        super(model_3, self).__init__()
        self.fea_extractor = model_3_sub()
        self.cls = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 80), nn.Sigmoid())
        self.att = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 80), nn.Sigmoid())
    
    def forward(self, image1, image2, image3):
        
        fea1 = self.fea_extractor(image1)
        fea2 = self.fea_extractor(image2)
        fea3 = self.fea_extractor(image3)
        
        fea1 = torch.unsqueeze(fea1, dim=1)
        fea2 = torch.unsqueeze(fea2, dim=1)
        fea3 = torch.unsqueeze(fea3, dim=1)
        
        x = torch.cat((fea1, fea2, fea3), dim=1)
        
        #Time distribution in different time dim
        x_reshape = x.contiguous().view(-1, x.size(-1))
        x_cls = self.cls(x_reshape)
        x_att = self.att(x_reshape)
        x_cls = x_cls.view(-1, x.size(1), x_cls.size(-1))
        x_att = x_att.view(-1, x.size(1), x_att.size(-1))
        att = x_att / torch.sum(x_att, dim=1, keepdim=True)
        #attention layers
        output = att * x_cls
        output = torch.sum(output, dim=1, keepdim=False)
        output = torch.clamp(output, 1e-8, 1)
        
        return output
        
class model_4(nn.Module):

    def __init__(self):
        super(model_4, self).__init__()
        self.prepreocess = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU())
        self.cnn = nn.Sequential(Gate_ConvBlock(3, 64),   #(N, 64, 64, 64)
                                 Gate_ConvBlock(64, 128), #(N, 128, 32, 32)
                                 Gate_ConvBlock(128, 256),#(N, 256, 16, 16)
                                 Gate_ConvBlock(256, 512),#(N, 512, 8, 8)
                                 Gate_ConvBlock(512, 512))#(N, 512, 4, 4)
        self.gru = nn.GRU(512*4, 512, batch_first=True, bidirectional=True) 
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 80))
        self._init_weights()
          
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        
        x = self.prepreocess(x)
        x = self.cnn(x)
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(-1)) #(N, 2048, 4)  
        x = x.permute((0, 2, 1)) #(N, 4, 2048)
        x, _ = self.gru(x) #(N, 4, 1024)
        x = self.relu(x)
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1) #(N, 1024)
        x = self.fc(x) #(N, 80)
        
        return x

def model1_predict(x_test, num_classes=80, folds=10, shift=128, batch_size=256):
    
    all_data = []
    all_idx = []
    
    for idx, data in enumerate(x_test):
        base_dim, time_dim = data.shape[0], data.shape[1]
        for j in range(((time_dim - base_dim) + 1) // shift + 1):
            start = j * shift
            start = min(start, time_dim-base_dim)
            end = start + base_dim
            end = min(end, time_dim)
            cropped = data[:, start:end, :]
            all_data.append(cropped)
            all_idx.append(idx)
        
    test_df_ = pd.DataFrame()
    test_df_['data'] = all_data
    test_df_['idx'] = all_idx
    
    test_dataset = Model1_FATTestDataset(test_df_, config.transforms_dict['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_preds = np.zeros((len(x_test),80))
    for fold in range(folds):
        
        model = model_1().cuda()
        model.load_state_dict(torch.load('../input/freesound-model/model_1/weight_best_{}.pt'.format(fold)))
        model.cuda()
        model.eval()
        
        all_outputs = []
        
        for images in test_loader:
            preds = torch.sigmoid(model(images.cuda()).detach())
            all_outputs.append(preds.cpu().numpy())
            
        test_preds_tmp = pd.DataFrame(data=np.concatenate(all_outputs),
                                      index=test_df_.idx,
                                      columns=map(str, range(num_classes)))
        test_preds_tmp = test_preds_tmp.groupby(level=0).mean()
        test_preds += test_preds_tmp.values
    
    test_preds /= 10
    
    return test_preds

def model2_predict(test_fnames, x_test, test_transforms=config.transforms_dict['test'], num_classes=80, folds=10, tta=5):
    
    all_data = []
    all_idx = []
    
    batch_size = 64
    test_dataset = Model2_FATTestDataset(test_fnames, x_test, test_transforms, tta=tta)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_preds   = np.zeros((len(x_test),80))
    
    for fold in range(folds):
        
        model = model_2().cuda()
        model.load_state_dict(torch.load('../input/freesound-model/model_2/weight_best_{}.pt'.format(fold)))
        model.cuda()
        model.eval()
        
        all_outputs, all_fnames = [], []
        
        for inputs, fnames in test_loader:
            image1, image2 = inputs
            image1, image2 = image1.cuda(), image2.cuda()
            preds = model(image1, image2).detach()
            all_outputs.append(preds.cpu().numpy())
            all_fnames.extend(fnames)

        test_preds_tmp = pd.DataFrame(data=np.concatenate(all_outputs),
                                      index=all_fnames,
                                      columns=map(str, range(num_classes)))
        test_preds_tmp = test_preds_tmp.groupby(level=0).mean()
        test_preds += test_preds_tmp.values
    
    test_preds /= 10
    
    return test_preds

def model3_predict(test_fnames, x_test, test_transforms=config.transforms_dict['test'], num_classes=80, folds=10, tta=5):
    
    all_data = []
    all_idx = []
    
    batch_size = 32
    test_dataset = Model3_FATTestDataset(test_fnames, x_test, test_transforms, tta=tta)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_preds   = np.zeros((len(x_test),80))
    
    for fold in range(folds):
        
        model = model_3().cuda()
        model.load_state_dict(torch.load('../input/freesound-model/model_3_1/weight_best_{}.pt'.format(fold)))
        model.cuda()
        model.eval()
        
        all_outputs, all_fnames = [], []
        
        for inputs, fnames in test_loader:
            image1, image2, image3 = inputs
            image1, image2, image3 = image1.cuda(), image2.cuda(), image3.cuda()
            preds = model(image1, image2, image3).detach()
            all_outputs.append(preds.cpu().numpy())
            all_fnames.extend(fnames)
            
        test_preds_tmp = pd.DataFrame(data=np.concatenate(all_outputs),
                                      index=all_fnames,
                                      columns=map(str, range(num_classes)))
        test_preds_tmp = test_preds_tmp.groupby(level=0).mean()
        test_preds += test_preds_tmp.values
    
    test_preds /= 10
    
    return test_preds
    
def model4_predict(test_fnames, x_test, test_transforms=config.transforms_dict['test'], num_classes=80, folds=10, tta=5):
    
    all_data = []
    all_idx = []
    
    batch_size = 64
    test_dataset = Model4_FATTestDataset(test_fnames, x_test, test_transforms, tta=tta)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_preds   = np.zeros((len(x_test),80))
    
    for fold in range(folds):
        
        model = model_4().cuda()
        model.load_state_dict(torch.load('../input/freesound-model/model_4/weight_best_{}.pt'.format(fold)))
        model.cuda()
        model.eval()
        
        all_outputs, all_fnames = [], []
        
        for images, fnames in test_loader:
            preds = torch.sigmoid(model(images.cuda()).detach())
            all_outputs.append(preds.cpu().numpy())
            all_fnames.extend(fnames)

        test_preds_tmp = pd.DataFrame(data=np.concatenate(all_outputs),
                                      index=all_fnames,
                                      columns=map(str, range(num_classes)))
        test_preds_tmp = test_preds_tmp.groupby(level=0).mean()
        test_preds += test_preds_tmp.values
    
    test_preds /= 10
    
    return test_preds
    
if __name__ == '__main__':
    
    seed_everything(config.SEED)
    test_df = pd.read_csv(config.csvs['sample_submission'])
    create_mel_inputs(test_df, 'test', check=False)
    with open('../working/test.pickle', 'rb') as test:
        x_test = pickle.load(test)
        
   
    labels = test_df.columns[1:].tolist()
    
    print('model 1 prediction start..')
    start_time = time.time()
    model1_test_preds = model1_predict(x_test) # stage1 -> 0.683
    print('model 1 prediction end.., spend {}s'.format(time.time()-start_time))
    print('model 3 prediction start..')
    start_time = time.time()
    model3_test_preds = model3_predict(test_df['fname'], x_test, tta=25) #stage1-> 0.694
    print('model 3 prediction end.., spend {}s'.format(time.time()-start_time))
    print('model 4 prediction start..')
    start_time = time.time()
    model4_test_preds = model4_predict(test_df['fname'], x_test, tta=25) # Stage1 -> 0.690
    print('model 4 prediction end.., spend {}s'.format(time.time()-start_time))
    
    test_df[labels] = 0.1*model1_test_preds + 0.5*model3_test_preds + 0.4*model4_test_preds 
    test_df.to_csv('submission.csv', index=False)
    
    
    
    
    