from sklearn.model_selection import StratifiedKFold
import random
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

import config
from data import *
from model import *
from preprocess import *
from utils import *
from trainer import run_training, run_training_v2


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def bms_collate(batch):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)


def train_model():
    train = pd.read_pickle('train2.pkl')
    folds = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
    for n, (train_index, val_index) in enumerate(folds.split(train, train['InChI_length'])):
        train_df = train.iloc[train_index]
        val_df = train.iloc[val_index]
        config.valid_labels = val_df['InChI'].values
        print(f"train data size :{len(train_df)}")
        print(f"val data size :{len(val_df)}")
        train_dataset = TrainDataset(train_df, tokenizer=tokenizer,
                                     transform=get_transforms(config, mode="train"))

        validate_dataset = TestDataset(val_df, transform=get_transforms(config, mode="valid"))

        encoder_model = Encoder(model_name=config.encoder_model_name,
                                pretrained=config.encoder_model_pretrained)
        decoder_model = Decoder(vocab_size=len(tokenizer.stoi),
                                num_dims=config.decoder_dim,
                                max_len=config.max_len,
                                num_headers=config.num_head,
                                num_layer=config.num_layer)
        run_training(encoder_net=encoder_model, decoder_net=decoder_model,
                     train_dataset=train_dataset, validation_dataset=validate_dataset,
                     tokenizer=tokenizer, collate_fn=bms_collate, config=config)


def train_model_v2():
    train = pd.read_pickle('train2.pkl')
    folds = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
    for n, (train_index, val_index) in enumerate(folds.split(train, train['InChI_length'])):
        config.fold_num = n
        train_df = train.iloc[train_index]
        val_df = train.iloc[val_index]
        config.valid_labels = val_df['InChI'].values
        print(f"train data size :{len(train_df)}")
        print(f"val data size :{len(val_df)}")
        train_dataset = TrainDataset(train_df, tokenizer=tokenizer,
                                     transform=get_transforms(config, mode="train"))

        validate_dataset = TestDataset(val_df, transform=get_transforms(config, mode="valid"))

        # encoder_model = Encoder(model_name=config.encoder_model_name,
        #                         pretrained=config.encoder_model_pretrained)
        # decoder_model = Decoder(vocab_size=len(tokenizer.stoi),
        #                         num_dims=config.decoder_dim,
        #                         max_len=config.max_len,
        #                         num_headers=config.num_head,
        #                         num_layer=config.num_layer)
        run_training_v2(model=InchiModel(vocab_size=len(tokenizer.stoi)),
                        train_dataset=train_dataset, validation_dataset=validate_dataset,
                        tokenizer=tokenizer, collate_fn=bms_collate, config=config)

        break

if __name__ == '__main__':
    import os
    seed_everything(config.seed)
    if os.path.exists('tokenizer.pth') and config.recreate_tokenizer is False:
        tokenizer = torch.load('tokenizer.pth')
        print(f"tokenizer.stoi: {tokenizer.stoi}")
    else:
        train_csv = pd.read_csv(os.path.join(config.train_csv_directory, "train_labels.csv"))
        train_csv['InChI_1'] = train_csv['InChI'].apply(lambda x: x.split('/')[1])
        train_csv['InChI_text'] = train_csv['InChI_1'].apply(split_form) + ' ' + \
                                  train_csv['InChI'].apply(lambda x: '/'.join(x.split('/')[2:])).apply(
                                      split_form2).values
        train_csv['file_path'] = train_csv['image_id'].apply(get_train_file_path)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_csv['InChI_text'].values)
        torch.save(tokenizer, 'tokenizer.pth')

    config.pad_index = tokenizer.stoi['<pad>']

    train_model_v2()
