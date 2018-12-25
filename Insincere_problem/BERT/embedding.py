from keras.engine import Layer
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.layers import Embedding

class EmbeddingLayer(Layer):

    def __init__(self, vocab_size, embed_size, embedding_table, **kwargs):

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding_table = embedding_table
        super(EmbeddingLayer, self).__init__(**kwargs)

    def call(self, x):

        x = K.cast(x, dtype='int32')
        one_hot_inputs_ids = K.one_hot(x, self.vocab_size)
        output = K.dot(one_hot_inputs_ids, self.embedding_table)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embed_size)
