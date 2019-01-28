from keras.layers import *
from keras.models import *
from ConvSelfAttention import *
from PositionEmbedding import *
from LayerNormarlization import *

class HCAN(object):

    def __init__(self, max_len, n_filter=512,
                 max_features=50000, embed_size=512,
                 kernel_size=3):

        self.max_len      = max_len
        self.n_filter     = n_filter
        self.kernel_size  = kernel_size
        self.max_features = max_features
        self.embed_size   = embed_size

        #assert len(kernel_size) == 3

    def build(self):

        inp = Input((self.max_len,))
        x = Embedding(self.max_features, self.embed_size)(inp)
        x = PositionEmbedding(self.max_len, self.embed_size, mode='Concat')(x)

        Q_a = Conv1D(self.n_filter, kernel_size=self.kernel_size, activation='elu')(x)
        K_a = Conv1D(self.n_filter, kernel_size=self.kernel_size, activation='elu')(x)
        V_a = Conv1D(self.n_filter, kernel_size=self.kernel_size, activation='elu')(x)

        Q_b = Conv1D(self.n_filter, kernel_size=self.kernel_size, activation='elu')(x)
        K_b = Conv1D(self.n_filter, kernel_size=self.kernel_size, activation='elu')(x)
        V_b = Conv1D(self.n_filter, kernel_size=self.kernel_size, activation='tanh')(x)

        MultiHead_a = ConvSelfAttention(8, self.n_filter)([Q_a, K_a, V_a])
        MultiHead_b = ConvSelfAttention(8, self.n_filter)([Q_b, K_b, V_b])

        x = Lambda(lambda x: x[0] * x[1])([MultiHead_a, MultiHead_b])
        x = LayerNormalization()(x)

        return Model(inp, x)

model = HCAN(max_len=100).build()
model.summary()
