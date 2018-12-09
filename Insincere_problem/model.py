#"Convolutional Neural Networks for Sentence Classification"
#https://arxiv.org/pdf/1408.5882.pdf
#"Character-level Convolutional Networks for Text Classification∗"
#https://www.kaggle.com/yekenot/2dcnn-textclassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.layers import Input, Conv1D, Embedding, MaxPool1D, Dense, Dropout, Flatten, Concatenate
from keras.optimizers import adam
from keras.models import Model

class create_model(object):

    def __init__(self, num_filter, Ouput_Unit=None, kernel_size=None, max_features=50000, max_len=100, embed_size=300, pretrain_embeding_matrix=None):

        self.max_features = max_features
        self.max_len = max_len
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.Ouput_Unit = Ouput_Unit
        self.num_filter = num_filter
        self.pretrain_embed_matrix = pretrain_embeding_matrix
        self.adam = adam(lr=0.01, decay=1e-6)

    def get_cnn(self, check=False):

        """
        A sentence of length n (padded where necessary) is represented as
        x1:n = x1 ⊕ x2 ⊕ . . . ⊕ xn
        where ⊕ is the concatenation operator.

        A convolution operation involves a filter w ∈ R hk, which is applied to a window
        of h words to produce a new feature. For example, a feature ci is generated
        from a window of words xi:i+h−1 by
        ci = f(w · xi:i+h−1 + b).

        This filter is applied to each possible window of words in the sentence.
        The model uses multiple filters, (with varying window sizes)
        loss are passed to a fully connected softmax layer whose output is the probability
        distribution over labels.

        More discussion:
        While we had expected performance gains through the use of pre-trained vectors,
        we were surprised at the magnitude of the gains.

        """

        inputs = Input(shape=(self.max_len,))
        x = Embedding(self.max_features, self.embed_size, weights=self.pretrain_embed_matrix)(inputs) # shape: batch_size, max_len, emb_size

        #create feature map, to each possible window
        #c = [c1, c2, ..., cn−h + 1]

        conv_0 = Conv1D(filters=self.num_filter, kernel_size=(self.kernel_size[0]),
                        kernel_initializer='he_normal', activation='elu')(x) # (batch_size, max_len-kernel_size + 1 , num_filter)
        conv_1 = Conv1D(filters=self.num_filter, kernel_size=(self.kernel_size[1]),
                        kernel_initializer='he_normal', activation='elu')(x) #  (batch_size, max_len-kernel_size + 1 , num_filter)
        conv_2 = Conv1D(filters=self.num_filter, kernel_size=(self.kernel_size[2]),
                        kernel_initializer='he_normal', activation='elu')(x)
        conv_3 = Conv1D(filters=self.num_filter, kernel_size=(self.kernel_size[3]),
                        kernel_initializer='he_normal', activation='elu')(x)

        maxpool_0 = MaxPool1D(pool_size=(self.max_len - self.kernel_size[0] + 1))(conv_0)
        maxpool_1 = MaxPool1D(pool_size=(self.max_len - self.kernel_size[1] + 1))(conv_1)
        maxpool_2 = MaxPool1D(pool_size=(self.max_len - self.kernel_size[2] + 1))(conv_2)
        maxpool_3 = MaxPool1D(pool_size=(self.max_len - self.kernel_size[3] + 1))(conv_3)

        # Fully connected layer with dropout and softmax output

        z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        z = Flatten()(z)
        #dropout on the penultimate layer with a constraint on l2-norms of the weight vectors(Hintonetal., 2012).
        z = Dropout(0.1)(z)
        outputs = Dense(1, activation='sigmoid', use_bias=True)(z)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

        if check:
            print(model.summary())

        return model

    def get_chara_level_Net(self, check=False):

        inputs = Input(shape=(self.max_len,))
        x = Embedding(self.max_features, self.embed_size, weights=self.pretrain_embed_matrix)(inputs)  # shape: batch_size, max_len, emb_size
        net = Conv1D(filters=self.num_filter, kernel_size=(7),
                        kernel_initializer='he_normal', activation='elu')(x)
        net = MaxPool1D(pool_size=(3))(net)
        net = Conv1D(filters=self.num_filter, kernel_size=(7),
                        kernel_initializer='he_normal', activation='elu')(net)
        net = MaxPool1D(pool_size=(3))(net)
        net = Conv1D(filters=self.num_filter, kernel_size=(3),
                        kernel_initializer='he_normal', activation='elu')(net)
        net = Conv1D(filters=self.num_filter, kernel_size=(3),
                     kernel_initializer='he_normal', activation='elu')(net)
        net = MaxPool1D(pool_size=(3))(net)

        net = Flatten()(net)
        net = Dense(self.Ouput_Unit, activation='elu', use_bias=True)(net)
        net = Dense(self.Ouput_Unit, activation='elu', use_bias=True)(net)
        outputs = Dense(1, activation='sigmoid', use_bias=False)(net)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

        if check:
            print(model.summary())

        return model


model = create_model(num_filter=256, Ouput_Unit=2048).get_chara_level_Net(check=True)

