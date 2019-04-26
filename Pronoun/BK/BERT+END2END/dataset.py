from torch.utils.data import Dataset
from torchvision import transforms
from ast import literal_eval

class MyDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.csv = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        index_token = self.csv.loc[idx, 'indexed_token']
        index_token = literal_eval(index_token) # Change string to list
        index_token = pad_sequences([index_token], maxlen=100, padding='post')[0]

        offset = self.csv.loc[idx, 'offset']
        offset = literal_eval(offset)
        offset = np.asarray(offset, dtype='int32')
        label  = int(self.csv.loc[idx, 'label'])

        if self.transform:
            index_token = self.transform(index_token)
            offset = self.transform(offset)
            label = self.transform(label)

        return index_token, offset, label

trainset = MyDataset('../working/test.csv')
train_loader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)
#for x, (index_token, offset, labels) in enumerate(train_loader):
