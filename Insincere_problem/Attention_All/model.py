import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('./')
from keras.layers import Input, Embedding, Dense, Flatten, GlobalAvgPool1D
from keras.optimizers import adam
from keras.models import Model
from utils import Position_Embedding, MultiHeadAttention
import os
os.chdir(os.path.dirname(__file__))


class create_model(object):

    def __init__(self, max_features=50000, max_len=100, embed_size=300, pretrain_embeding_matrix=None):

        self.max_features = max_features
        self.max_len = max_len
        self.embed_size = embed_size
        self.pretrain_embed_matrix = pretrain_embeding_matrix
        self.adam = adam(lr=0.01, beta_1=0.9, beta_2=0.98, decay=1e-9)

    def get(self):

        inputs = Input(shape=(self.max_len,))
        x = Embedding(self.max_features, self.embed_size)(inputs)
        x = Position_Embedding()(x)

        O_seq = MultiHeadAttention(8, 16)([x, x, x])
        O_seq = GlobalAvgPool1D()(O_seq)
        Output = Dense(1, activation='sigmoid')(O_seq)

        model = Model(inputs=inputs, outputs=Output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model = create_model(max_features=max_features, max_len=maxlen).get()
model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
