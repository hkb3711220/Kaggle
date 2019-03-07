from keras.layers import *
from keras.models import Model, Sequential
import numpy as np

class MentionPairEncoder(object):

    def __init__(self, input_shape1, input_shape2, hidden_dim1=300,
                 hidden_dim2=150, output_dim=3, drop_rate=0.5):

        self.num_feas1, self.feas1_dims = input_shape1[1], input_shape1[2]
        self.num_feas2 = input_shape2[1]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.drop_rate = drop_rate
        self.output_dim = output_dim

    def build(self):

        inpM = Input(shape=(self.num_feas1, self.feas1_dims)) #mention ex. His
        inpA = Input(shape=(self.num_feas1, self.feas1_dims)) #Candidate Antecedent A ex. Bob Suter
        inpB = Input(shape=(self.num_feas1, self.feas1_dims)) #Candidate Antecedent B ex Dehner
        input1 = [inpM, inpA, inpB]

        inpM_A = Input(shape=(self.num_feas2,)) #Mention with Antecedent A
        inpM_B = Input(shape=(self.num_feas2,)) #Mention with Antecedent B
        input2 = [inpM_A, inpM_B]

        Feature_map_Layer = Dense(self.feas1_dims, activation='relu')
        reshape_layer = Reshape((1, self.feas1_dims))

        x2 = [Feature_map_Layer(_input2) for _input2 in input2]
        x2 = [reshape_layer(_x2) for _x2 in x2]

        pair1 = Concatenate(axis=1)([input1[0], input1[1], x2[0]]) # all of those featues are concatenated to produce an I-dimention vector h0
        pair2 = Concatenate(axis=1)([input1[0], input1[2], x2[1]])

        M_rep1 = self.hidden_layers(pair1)
        M_rep2 = self.hidden_layers(pair2)

        x = Concatenate(axis=-1, name="cluster-poir")([M_rep1, M_rep2]) # cluster-poir, shape(batch_size, num_feas1+1, 2*hidden_dim2)
        x = TimeDistributed(Dropout(rate=self.drop_rate))(x)
        x = TimeDistributed(Dense(10, activation='relu', use_bias=True))(x)
        x = Flatten()(x)

        x = BatchNormalization()(x)
        x = Dropout(rate=self.drop_rate)(x)
        output = Dense(self.output_dim, activation='softmax')(x)

        model = Model(input1+input2, output)

        return model

    def hidden_layers(self, x):

        x = Dense(self.hidden_dim1, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(self.hidden_dim2, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(self.hidden_dim2, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)

        return x

#model = MentionPairEncoder(input_shape1=(64,32,300), input_shape2=(64, 45)).build()
#model.summary()
