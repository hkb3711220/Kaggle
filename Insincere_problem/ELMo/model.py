#For test
from ELMo_Keras import ELMoEmbeddingLayer
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv1D, Embedding, MaxPool1D, Dense, Dropout, Flatten, Concatenate
from keras.optimizers import adam
from keras.models import Model

# Initialize session
sess = tf.Session()
K.set_session(sess)

class create_model(object):

    def __init__(self):
        self.adam = adam(lr=0.01)

    def get(self):

        """
        To Add ELMo to the supervised model,
        concatenate thE ELMo vector ELMok with xk and pass ElMo enhanced representation [xk;ELMok] into task RNN.

        """

        inputs = Input(shape=(1,), dtype='string')
        embedding = ELMoEmbeddingLayer()(inputs)
        net = Dense(256, activation='relu')(embedding)
        outputs = Dense(1, activation='sigmoid')(net)

        model = Model(inputs=[inputs], outputs=outputs)
        model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])


        return model

model = create_model().get()
model.summary()
