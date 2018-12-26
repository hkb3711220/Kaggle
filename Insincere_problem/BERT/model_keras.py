import sys
sys.path.append('./')

from keras.layers import Input, Embedding, Dense, Lambda, add
from keras.optimizers import adam
from keras.models import Model
from keras import backend as K
from keras.initializers import truncated_normal
from utils import Position_Embedding, LayerNormalization
from embedding import EmbeddingLayer
from attention import AttentionLayer
import numpy as np
import tensorflow as tf
from keras.utils import plot_model


class create_model(object):

    def __init__(self, num_class=2, num_layer=12, vocab_size=200, max_len=3, embed_size=768, one_hot=False):

        self.num_class = num_class
        self.num_layer = num_layer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_size = embed_size
        self.one_hot = one_hot
        self.embedding_table = K.truncated_normal(shape=(self.vocab_size, self.embed_size), stddev=0.02)
        self.initializers = truncated_normal(stddev=0.02)

    def build(self):

        inputs = Input(shape=(self.max_len,))

        # Because my input only have one sentence
        # the embedding layers(exclude segment embedding)
        # the position embedding is from <<Attention of All you Need>>
        if self.one_hot:
            x = EmbeddingLayer(self.vocab_size, self.embed_size, self.embedding_table)(inputs)
        else:
            x = Embedding(self.vocab_size, self.embed_size, embeddings_initializer=self.initializers)(inputs)

        layer_input = Position_Embedding()(x)
        prev_output = layer_input

        self.all_layer_outputs = []

        for num in range(self.num_layer):
            with tf.name_scope('transformer_block_{}'.format(num)):
                attention_output = AttentionLayer(num_attention_heads=12, output_dim=self.embed_size)(prev_output)
                attention_output = add([attention_output, prev_output])
                attention_output = LayerNormalization()(attention_output)

                intermediate_output = Dense(self.embed_size*4, activation='elu', kernel_initializer=self.initializers)(attention_output)
                layer_output = Dense(self.embed_size, kernel_initializer=self.initializers)(intermediate_output)
                layer_output = add([layer_output, attention_output])
                layer_output = LayerNormalization()(layer_output)
                self.all_layer_outputs.append(layer_output)
                prev_output = layer_output

        self.sequence_output = prev_output
        self.frist_token = Lambda(lambda x: K.squeeze(x[:,0:1,:], axis=1))(self.sequence_output)
        self.pooled_out = Dense(units=self.num_class, activation='softmax')(self.frist_token)

        model = Model(inputs=inputs, outputs=self.pooled_out)

        return model

    def get_all_layer_outputs(self):
        return self.all_layer_outputs

    def get_first_token(self):
        return self.frist_token

    def get_pooled_out(self):
        return self.pooled_out

model = create_model(one_hot=True).build()
model.summary()
plot_model(model, to_file='model.png')
#model.compile(optimizer=adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
x = np.asarray([[10, 20, 3], [3, 34, 0]])
#print(x.shape)

output = model.predict(x=x)
print(output)
