import os
import pandas as pd
from model import create_model
from sklearn.model_selection import train_test_split

path = r'C:\Users\user1\Desktop\Kaggle\Insincere_problem\Input'

csv_path_train = os.path.join(path, 'train.csv')
csv_data_train = pd.read_csv(csv_path_train)
train_data, val_data = train_test_split(csv_data_train, test_size=0.1, random_state=2018)

train_X = train_data["question_text"].fillna("_na_").values
val_X = val_data["question_text"].fillna("_na_").values
train_y = train_data['target'].values
val_y = val_data['target'].values

csv_path_test = os.path.join(path, 'test.csv')
csv_data_test = pd.read_csv(csv_path_test)
test_X = csv_data_test["question_text"].fillna("_na_").values
test_qid = csv_data_test['qid']

model = create_model().get()
model.summary()

if __name__=='__main__':
    model.fit(train_X, train_y, batch_size=128, epochs=2, validation_data=(val_X, val_y))
