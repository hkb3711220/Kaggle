import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement
import numpy.random as rng

os.chdir(os.path.dirname(__file__))
data = pd.read_csv("./input/train.csv")
categories = data.groupby('Id').size()
batch_size = 32
base_size = 105

# Make Categories dictionary
Id_dict = {}
for i, id in enumerate(categories.index):
    if id in Id_dict:
        continue
    Id_dict[id] = i

categories_list = []
for k in Id_dict.keys():
    img_list = data[data.Id == k].Image.values
    categories_list.append(img_list)

def read_image(path, image_name, base_size):

    image_path = os.path.join(path, image_name)
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (base_size, base_size))
    img = img / 255

    return img

def select_img(img_list):

    if len(img_list) > 1:
        choice_index = rng.randint(len(img_list), size=1)
        choice_img = img_list[choice_index][0]
    else:
        choice_img = img_list[0]

    return  choice_img

def make_batch(batch_size):

    img1 = []
    img2 = []

    categories = rng.choice(len(Id_dict), size=(batch_size,),replace=False)
    label = np.zeros(batch_size)
    label[batch_size//2:] = 1.0

    for i in range(batch_size):
        category_num = categories[i]
        img_list = categories_list[category_num]
        x1 = select_img(img_list)

        if i >= batch_size // 2:
            dif_num = rng.randint(len(Id_dict), size=1)
            while dif_num == category_num:
                dif_num = rng.randint(len(Id_dict), size=1)

            dif_img_list = categories_list[int(dif_num)]
            x2 = select_img(dif_img_list)
        else:
            x2 = select_img(img_list)

        img1.append(x1)
        img2.append(x2)

    return img1, img2, label

def generator(data, batch_size):

    while True:
        img1, img2, label = make_batch(batch_size=batch_size)
        left_input = np.zeros((len(img1), base_size, base_size, 1))
        right_input = np.zeros((len(img2), base_size, base_size, 1))

        for i, img_name in enumerate(img1):
            left_input[i, :, :, 0] = read_image(path='../input/train', image_name=img_name, base_size=base_size)
        for i, img_name in enumerate(img2):
            right_input[i, :, :, 0] = read_image(path='../input/train', image_name=img_name, base_size=base_size)

        yield [left_input, right_input], label
