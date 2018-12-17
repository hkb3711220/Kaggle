import tensorflow_hub as hub
import tensorflow as tf

class ELMoEmbeddingLayer(object):

    def __init__(self):
        self.dimension = 1024
        self.trainable = True

    def call(self, x, mode='default'):

        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable,
                               name="{}_module".format('ELMo'))

        if mode == 'default' :
            embedding = self.elmo(x, signature=mode, as_dict=True)['elmo']

        elif mode == 'tokens':
            """
            the input should be a turple
            """
            tokens_inputs = x[0]
            tokens_length = x[1]
            embedding = self.elmo(input={'tokens': tokens_inputs,
                                         'sequence_len': tokens_length},
                                         signature=mode,
                                         as_dict=True)['elmo']

        return embedding

#For test
#x = tf.placeholder(tf.string, shape=[None,])
#embeddings = ELMoEmbeddingLayer().call(x)

#with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.tables_initializer())

    #print(sess.run(embeddings, feed_dict={x:["the cat is on the mat", "dogs are in the fog"]}))
