#https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440i
import tensorflow_hub as hub
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class ELMoEmbeddingLayer(Layer):

    def __init__(self, **kwargs):

        self.dimension = 1024
        self.trainable = True
        super(ELMoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format('self.name'))

        super(ELMoEmbeddingLayer, self).build(input_shape)

    def call(self, x):
        """
        1.With the default signature, the module takes untokenized sentences as input.
        The input tensor is a string tensor with shape [batch_size]. The module tokenizes each string by splitting on spaces.
        2.With the tokens signature, the module takes tokenized sentences as input.
        The input tensor is a string tensor with shape [batch_size, max_length] and an int32 tensor with shape [batch_size] corresponding to the sentence length.
        The length input is necessary to exclude padding in the case of sentences with varying length.

        """
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           signature='default', as_dict=True)['default']

        return result

    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.dimension)
