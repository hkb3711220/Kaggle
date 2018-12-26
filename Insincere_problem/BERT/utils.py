from keras import backend as K
from keras.engine.topology import Layer


class Position_Embedding(Layer):

    def __init__(self, size=None, **kwargs):

       """
       "positional encodings" to the input embeddings at the
       bottoms of the encoder and decoder stacks.

       """
       self.size = size
       super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):

        if self.size == None:
            self.size = int(x.shape[-1]) #embed_size
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2*K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1) -1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.sin(position_ij), K.cos(position_ij)], axis=2)
        position_embeding = position_ij + x

        return position_embeding #batch_size, max_len, embed_size

    def compute_output_shape(self, input_shape):
        return input_shape

class LayerNormalization(Layer):

    def __init__(self, eps=1e-6, **kwargs):

        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma  = self.add_weight(name='gamma',
                                     shape=(input_shape[-1:]),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.beta = self.add_weight(name='bata',
                                  shape=(input_shape[-1:]),
                                  initializer='glorot_uniform',
                                  trainable=True)

        super(LayerNormalization,self).build(input_shape)

    def call(self, x):
        means = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        x_norm = (x - means) / ((std + self.eps)**0.5)
        output = self.gamma * x_norm + self.beta

        return output

    def compute_output_shape(self, input_shape):
        return input_shape
