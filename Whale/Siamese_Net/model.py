import keras.backend as K
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Input, merge, subtract, Lambda
from keras.models import Sequential, Model
from keras.initializers import random_normal

class Siamese_Net(object):

    def __init__(self, input_shape):

        self.input_shape = input_shape
        self.initializers_weight = random_normal(mean=0.0, stddev=0.01)
        self.initializers_bias = random_normal(mean=0.5, stddev=0.01)
        self.initializers_weight_fully = random_normal(mean=0.0, stddev=0.2)

    def build(self):

        left_input = Input(self.input_shape)
        right_input = Input(self.input_shape)

        convert = Sequential()
        convert.add(Conv2D(64, kernel_size=10, strides=(1,1), activation='relu',
                           kernel_initializer=self.initializers_weight, bias_initializer=self.initializers_bias))
        convert.add(MaxPool2D(pool_size=2))
        convert.add(Conv2D(128, kernel_size=7, strides=(1,1), activation='relu',
                           kernel_initializer=self.initializers_weight, bias_initializer=self.initializers_bias))
        convert.add(MaxPool2D(pool_size=2))
        convert.add(Conv2D(128, kernel_size=4, strides=(1,1), activation='relu',
                           kernel_initializer=self.initializers_weight, bias_initializer=self.initializers_bias))
        convert.add(MaxPool2D(pool_size=2))
        convert.add(Conv2D(256, kernel_size=4, strides=(1,1), activation='relu',
                           kernel_initializer=self.initializers_weight, bias_initializer=self.initializers_bias))
        #the units in the final convolutional layer are flattened into a single vector
        convert.add(Flatten())
        #the convolutional layer is followed by a fully connected layer
        convert.add(Dense(4096, activation='sigmoid', kernel_initializer=self.initializers_weight_fully,
                          bias_initializer=self.initializers_bias))

        left_features = convert(left_input)
        right_features = convert(right_input)

        #L1 siamese dist
        dist = Lambda(lambda x: K.abs(x[0]-x[1]))([left_features, right_features])

        #fully connected + sigmoid
        out = Dense(1, activation='sigmoid')(dist)

        model = Model(inputs=[left_input, right_input], outputs=out)

        return model

model = Siamese_Net(input_shape=(105, 105, 1)).build()
model.summary()
