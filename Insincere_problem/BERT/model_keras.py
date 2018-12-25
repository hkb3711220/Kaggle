import sys
sys.path.append('./')

from keras.layers import Input, Embedding, Dense, add, Conv1D
from keras.optimizers import adam
from keras.models import Model
from keras import backend as K
from keras.initializers import truncated_normal
from utils import Position_Embedding, MultiHeadAttention, LayerNormalization
from embedding import EmbeddingLayer
import numpy as np

class create_model(object):

    def __init__(self, num_layer=12, vocab_size=200, max_len=3, embed_size=10, one_hot=False):

        self.num_layer = num_layer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_size = embed_size
        self.one_hot = one_hot
        self.embedding_table = K.truncated_normal(shape=(self.vocab_size, self.embed_size), stddev=0.02)
        self.initializers = truncated_normal(stddev=0.02)

    def get(self):

        inputs = Input(shape=(self.max_len,))

        # the embedding layers(exclude segment embedding)
        # the position embedding is from <<Attention of All you Need>>
        if self.one_hot:
            x = EmbeddingLayer(self.vocab_size, self.embed_size, self.embedding_table)(inputs)
        else:
            x = Embedding(self.vocab_size, self.embed_size, embeddings_initializer=self.initializers)(inputs)
        layer_input = Position_Embedding()(x)

        model = Model(inputs=inputs, outputs=layer_input)

        return model



model = create_model(one_hot=True).get()
model.summary()
#model.compile(optimizer=adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
x = np.asarray([[10, 20, 3], [3, 34, 0]])
print(x.shape)
output = model.predict(x=x)
print(output)
