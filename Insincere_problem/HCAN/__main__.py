from keras.layers import *
from keras.models import *
from ConvSelfAttention import *
from PositionEmbedding import *
from LayerNormarlization import *
from ConvTargetAttention import *

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
        x   = Embedding(self.max_features, self.embed_size)(inp)
        x   = PositionEmbedding(self.max_len, self.embed_size)(x)

        a_1 = ConvSelfAttention(kernel_size=3, n_head=8, output_dim=self.embed_size)(x)
        a_2 = ConvSelfAttention(kernel_size=3, n_head=8, output_dim=self.embed_size, V_activation='tanh')(x)
        x   = Lambda(lambda x: x[0] * x[1])([a_1, a_2])
        x   = LayerNormalization()(x)

        T   = GlobalAvgPool1D()(x)
        x   = ConvTargetAttention(kernel_size=3, n_head=8, output_dim=self.embed_size)([T, x, x])#regradless the length, represent as sentence

        return Model(inp, x)

model = HCAN(max_len=100).build()
model.summary()
