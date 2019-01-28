from keras import backend as K
from keras.engine.topology import Layer

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
