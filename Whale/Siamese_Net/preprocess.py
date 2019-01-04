import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement

os.chdir(os.path.dirname(__file__))
data = pd.read_csv("./input/train.csv")
train_data, test_data = train_test_split(data, test_size=0.3, random_state=2018)
train_data.to_csv('./working/train.csv', index=False)

def preprocess(data):

    img1 = []
    img2 = []
    label = []

    imgs = data.Image.values
    perm = combinations_with_replacement(imgs, 2)

    for p in perm:
        (x1, x2)= p
        if (data[data.Image == x1].Id.values == data[data.Image == x2].Id.values): y=1.0
        else:y=0
        img1.append(x1)
        img2.append(x2)
        label.append(y)

        if len(label) == 5000: break

    return left_input, right_input, label

def read_image(path, image_name, base_size):
    image_path = os.path.join(path, image_name)
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (base_size, base_size))
    img = img / 255

    return img

def generator(data, batch_size):
    base_size = 64
    while True:
        for df in pd.read_csv(data, chunksize=batch_size):
            img1, img2, label = preprocess(df)
            left_input = np.zeros((len(img1), base_size, base_size, 1))
            right_input = np.zeros((len(img2), base_size, base_size, 1))

            for i, img_name in enumerate(img1):
                left_input[i, :, :, 0] = read_image(path='../input/train', image_name=img_name, base_size=base_size)
            for i, img_name in enumerate(img2):
                right_input[i, :, :, 0] = read_image(path='../input/train', image_name=img_name, base_size=base_size)

            yield [left_input, right_input], label

gen = generator(data='./working/train.csv', batch_size=16)
[left_input, right_input], label = next(gen)

print(len(label))
