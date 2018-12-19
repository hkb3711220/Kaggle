#Character-Aware Neural Language Models

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Lambda, add, multiply, TimeDistributed, LSTM, Dropout, Flatten
from keras.models import Model
from keras import Sequential
from keras import backend as K
from keras.optimizers import SGD
from utils import DataLoader, generator


path = r'C:\Users\user1\Desktop\Kaggle\Insincere_problem\Input'

class embedding(object):

    def __init__(self,
                 num_filter,
                 kernel_size,
                 char_index,
                 max_features=35,
                 word_max_len=21,
                 embed_size=15,
                 highway=True):

        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.char_index = char_index
        self.max_features = max_features
        self.embed_size = embed_size
        self.word_max_len = word_max_len
        self.highway = highway

    def get(self):

        inputs = Input(shape=(self.word_max_len,))

        #Embedding
        x = Embedding(len(self.char_index)+1, self.embed_size, input_shape=(self.word_max_len,))(inputs) # shape: batch_size, max_len, emb_size

        feature_maps = []

        for i in range(len(self.kernel_size)):
            conv = Conv1D(filters=self.num_filter[i], kernel_size=(self.kernel_size[i]),
                          activation='tanh', name='conv_{}'.format(i))(x) # (batch_size, max_len-kernel_size + 1, num_filter)

            feature_maps.append(conv)

        max_pools=[]

        #Max over time pooling layer

        for i in range(len(feature_maps)):
            max_pool = GlobalMaxPooling1D(name='Maxovertimepoolinglayer_{}'.format(i))(feature_maps[i])
            max_pools.append(max_pool)

        #High way
        feature_vectors = Concatenate(axis=1)(max_pools)

        transform_gate = Dense(525, activation='sigmoid',name='transform_gate', use_bias=True)(feature_vectors)
        carry_gate = Lambda(lambda x: 1-x, name='carry_gate')(transform_gate)

        z = Dense(525, activation='relu')(feature_vectors)
        z = add([multiply([z, transform_gate]), multiply([carry_gate, feature_vectors])])

        model = Model(inputs=inputs, outputs=z)

        return model

class CharRNN(object):

    def __init__(self, num_unit, word_index):

        self.num_unit = num_unit
        self.max_len = 35
        self.word_index = word_index
        self.word_max_len = 21

    def get(self, embedding):

        model = Sequential()
        model.add(TimeDistributed(embedding, input_shape=(self.max_len, self.word_max_len)))
        model.add(LSTM(self.num_unit, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(self.num_unit, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(len(self.word_index), activation='linear')))

        return model

def PPL(y_true, y_pred):
    #エントロピー Hは，底 2 の負の対数尤度を単語数 |Wtest| で割った値で次式で表される．エントロピーが大きいほど次の単語の予測が困難（不確実）である．
    #パープレキシティ (Perplexity) PPL は， 2 のエントロピー乗．この値が小さいほど優れたモデルだといえる．
    return K.pow(2.0, K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=True)))

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred,from_logits=True)


if __name__ == '__main__':
    path = '.\Input'
    max_len = 35
    max_word_len = 21

    DataLoader_train = DataLoader(path, name='train')
    word_index, char_index = DataLoader_train.word_index, DataLoader_train.char_index
    inputs_tr, outputs_tr = DataLoader_train.inputs, DataLoader_train.outputs

    embedding = embedding(num_filter=[25, 50, 75, 100, 125, 150], kernel_size=[1, 2, 3, 4, 5 ,6], char_index=char_index).get()
    model = CharRNN(num_unit=300, word_index=word_index).get(embedding)
    model.summary()

    model.compile(optimizer=SGD(lr=1.0), loss=categorical_crossentropy, metrics=['accuracy', PPL])
    gen = generator(inputs_tr, outputs_tr, batch_size=20, max_len=max_len, max_word_len=max_word_len, word_index=word_index, char_index=char_index)

    #For test generator
    I = 0
    for x, y in gen:
        if I >= 10:
            break
        print("[Batch: {}]".format(I),"Input: ",x.shape,"Output:",y.shape)
        I += 1

    model.fit_generator(gen, steps_per_epoch=1267, epochs=25, verbose=1)
