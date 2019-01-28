from keras import backend as k
from keras.engine.topology import Layer

class ConvTargetAttention(Layer):

    """
    For classification purposes, we require that each sequence regardless
    of its length be represented by a single fixed-length vecotr V
    """

    def __init__(self, max_len, **kwargs):

        self.max_len = max_len
        super(ConvTargetAttention, self).__init__(**kwargs)

    def call(self, inputs):

        assert len(inputs) == 3
        T, K, V = inputs

        T_shape = k.shape(T) #⇒(batch_size, max_len, embed_size)
        K_shape = k.shape(K)
        V_shape = k.shape(V)

        heads = []
        for i in range(n_head):
            Ti = Q[:,i,:]
            Ki = K[:,i,:]
            Vi = V[:,i,:]

            a = k.batch_dot(Ti, Ki, axes=[2, 2]) / ((T_shape[2]) ** 0.5)
            a = k.softmax(a)
            a = k.batch_dot(a, Vi, axes=[2, 1])
            heads.append(a)

        heads = k.concatenate(heads)

        return heads
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2]


from keras.layers import *
from keras.models import *

x = Input(shape=(100,))
x_1 = Embedding(120000, 512)(x)
out = GlobalMaxPool1D()(x_1)

model = Model(x, out)
model = model.summary()
