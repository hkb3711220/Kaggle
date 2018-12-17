from keras.preprocessing.text import text_to_word_sequence
import os
import pandas as pd
from sklearn.model_selection import train_test_split

path = r'C:\Users\user1\Desktop\Kaggle\Insincere_problem\Input'

csv_path_train = os.path.join(path, 'train.csv')
csv_data_train = pd.read_csv(csv_path_train)
train_data, val_data = train_test_split(csv_data_train, test_size=0.1, random_state=2018)

train_X = train_data["question_text"].fillna("_na_").values
val_X = val_data["question_text"].fillna("_na_").values
train_y = train_data['target'].values
val_y = val_data['target'].values

train_X_tokens = []

for text in train_X:
    train_X_tokens.append([text_to_word_sequence(text)])

