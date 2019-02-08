from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class PositionEmbedding(Layer):

    def __init__(self, input_dim, output_dim,
                 mode='Add', **kwargs):

        self.input_dim   = input_dim #max_position_embeddings
        self.output_dim  = output_dim
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):

        self.embeddings = self.add_weight(name='position_matrix',
                                 shape=(self.input_dim, self.output_dim),
                                 initializer='random_uniform')
        super(PositionEmbedding, self).build(input_shape)

    def call(self, x):

        input_shape = K.shape(x)

        if self.mode == 'Add':
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim #concat

        #assert seq_len == self.input_dim
        position_embed = K.tile(K.expand_dims(self.embeddings[:seq_len, :], axis=0), K.stack([batch_size, 1, 1]))

        if self.mode == 'Add':
            return x + position_embed

        return K.concatenate([x, position_embed], axis=-1)
    def compute_output_shape(self, input_shape):
        if self.mode == 'Concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.output_dim)
        return input_shape
