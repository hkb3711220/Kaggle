import sys
sys.path.append('./')

from keras.layers import Input, Embedding, Dense, add, Conv1D
from keras.optimizers import adam
from keras.models import Model
from utils import Position_Embedding, MultiHeadAttention, LayerNormalization

class create_model(object):

    def __init__(self, num_layer=12, max_features=50000, max_len=100, embed_size=768, pretrain_embeding_matrix=None):

        self.num_layer=num_layer
        self.max_features = max_features
        self.max_len = max_len
        self.embed_size = embed_size
        self.pretrain_embed_matrix = pretrain_embeding_matrix

    def get(self):

        inputs = Input(shape=(self.max_len,))
        x = Embedding(self.max_features, self.embed_size)(inputs)
        prev_input = Position_Embedding()(x)

        for _ in range(self.num_layer):
            A = MultiHeadAttention(n_head=12)([prev_input, prev_input, prev_input])
            A = add([A, prev_input])
            A = LayerNormalization()(A)

            F = Conv1D(filters=self.embed_size * 4, kernel_size=(1), padding='same', activation='relu')(A)
            F = Conv1D(filters=self.embed_size, kernel_size=(1), padding='same')(F)
            F = add([A, F])
            prev_input = LayerNormalization()(F)

        FC = Dense(self.max_features, activation='sigmoid')(prev_input)

        model = Model(inputs=inputs, outputs=FC)

        return model

model = create_model().get()
model.summary()
model.compile(optimizer=adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
