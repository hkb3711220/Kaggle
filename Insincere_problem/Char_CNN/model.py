import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Lambda, add, multiply, TimeDistributed, LSTM, Dropout, Flatten
from keras.models import Model
from keras import Sequential
from keras import backend as K


class embedding(object):

    def __init__(self,
                 num_filter,
                 kernel_size,
                 max_features=30,
                 word_max_len=20,
                 embed_size=15,
                 highway=True):

        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.max_features = max_features
        self.embed_size = embed_size
        self.word_max_len = word_max_len
        self.highway = highway

    def get(self):

        inputs = Input(shape=(self.word_max_len,))
        x = Embedding(self.max_features, self.embed_size, input_shape=(self.word_max_len,))(inputs) # shape: batch_size, max_len, emb_size

        feature_maps = []

        for i in range(len(self.kernel_size)):
            conv = Conv1D(filters=self.num_filter[i], kernel_size=(self.kernel_size[i]),
                          activation='tanh', name='conv_{}'.format(i))(x) # (batch_size, max_len-kernel_size + 1, num_filter)

            feature_maps.append(conv)

        max_pools=[]

        for i in range(len(feature_maps)):
            max_pool = GlobalMaxPooling1D(name='Maxovertimepoolinglayer_{}'.format(i))(feature_maps[i])
            max_pools.append(max_pool)

        feature_vectors = Concatenate(axis=1)(max_pools)

        transform_gate = Dense(525, activation='sigmoid',name='transform_gate', use_bias=True)(feature_vectors)
        carry_gate = Lambda(lambda x: 1-x, name='carry_gate')(transform_gate)

        z = Dense(525, activation='relu')(feature_vectors)
        z = add([multiply([z, transform_gate]), multiply([carry_gate, feature_vectors])])

        model = Model(inputs=inputs, outputs=z)

        return model

class CharRNN(object):

    def __init__(self, num_unit):

        self.num_unit = num_unit
        self.max_len = 35
        self.word_max_len = 20

    def get(self, embedding):

        model = Sequential()
        model.add(TimeDistributed(embedding, input_shape=(self.max_len, self.word_max_len)))
        model.add(LSTM(self.num_unit, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(self.num_unit, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

embedding = embedding(num_filter=[25, 50, 75, 100, 125, 150], kernel_size=[1, 2, 3, 4, 5 ,6]).get()
model = CharRNN(num_unit=300).get(embedding)
model.summary()
