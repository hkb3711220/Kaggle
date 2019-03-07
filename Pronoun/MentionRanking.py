from keras.layers import *
from keras.models import Model, Sequential
import keras.backend as K

class MentionRanking(object):

    def __init__(self, input_shape1, input_shape2, hidden_dim1=100,
                 hidden_dim2=10, drop_rate=0.2):

        self.num_feas1, self.feas1_dims = input_shape1[1], input_shape1[2]
        self.num_feas2 = input_shape2[1]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.drop_rate = drop_rate

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

        pair1_score  = self.pair_layers(pair1) #(mention, AntecedentA)
        pair2_score  = self.pair_layers(pair2) #(mention, AntecedentB)
        single_score = self.single_layers(inpM) #(mention, NA)

        scores = Concatenate(axis=1)([pair1_score, pair2_score, single_score])
        output = Activation('softmax')(scores) #Apply a softmax over the socres for candidata antecedent
                                          #so probabilities sums to 1
        model = Model(input1+input2, output)

        return model

    def pair_layers(self, x):
        """
        The Mention-Ranking Model to a mention m and antecedent a
        representing their compatibility
        """

        x = Dense(self.hidden_dim1, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(self.hidden_dim2, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(self.hidden_dim2, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(1, activation='linear', use_bias=True)(x)
        x = Flatten()(x)
        x = Dense(1, activation='linear')(x)

        return x

    def single_layers(self, x):

        x = Dense(self.hidden_dim1, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(self.hidden_dim2, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(self.hidden_dim2, activation='relu', use_bias=True)(x)
        x = Dropout(rate=self.drop_rate)(x)
        x = Dense(1, activation='linear', use_bias=True)(x)
        x = Flatten()(x)
        x = Dense(1, activation='linear')(x)

        return x

model = MentionRanking(input_shape1=(64,11,300), input_shape2=(64, 45)).build()
model.summary()
