import keras.backend as K
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, Input, Concatenate, GlobalAvgPool2D, GlobalMaxPool2D, Lambda
from keras.models import Sequential, Model
from keras.initializers import random_normal
from keras.applications import NASNetMobile, DenseNet121

class Siamese_Net(object):

    def __init__(self, input_shape):

        self.input_shape = input_shape

    def build(self):

        left_input = Input(self.input_shape)
        right_input = Input(self.input_shape)

        convert = self.encode()
        left_features = convert(left_input)
        right_features = convert(right_input)

        #L1 siamese dist
        dist = Lambda(lambda x: K.abs(x[0]-x[1]))([left_features, right_features])

        #fully connected + sigmoid
        out = Dense(1, activation='sigmoid')(dist)

        model = Model(inputs=[left_input, right_input], outputs=out)

        return model

    def encode(self):

        input_tensor = Input(self.input_shape)

        base_model = DenseNet121(input_tensor=input_tensor, include_top=False, weights='imagenet')
        x = base_model.output
        out_1 = GlobalAvgPool2D()(x)
        out_2 = GlobalMaxPool2D()(x)
        out_3 = Flatten()(x)
        out = Concatenate(axis=-1)([out_1, out_2, out_3])
        out = Dropout(0.5)(out)
        out = Dense(4096, activation='sigmoid')(out)

        model = Model(inputs=input_tensor, outputs=out, name='encode_model')

        return model

model = Siamese_Net(input_shape=(64, 64, 3)).build()
model.summary()
