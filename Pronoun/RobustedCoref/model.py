import warnings
warnings.filterwarnings('ignore')
from keras.layers import *
from keras.models import *
import keras.backend as K
from keras.engine.topology import  Layer
import  tensorflow_hub as hub
import tensorflow as tf

class ELMoEmbeddingLayer(Layer):

    def __init__(self, max_len=None,  token=None,**kwargs):

        self.dimension = 1024
        self.max_len = max_len
        self.trainable = True
        self.token = token
        super(ELMoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))

        super(ELMoEmbeddingLayer, self).build(input_shape)

    def call(self, x):
        """
        1.With the default signature, the module takes untokenized sentences as input.
        The input tensor is a string tensor with shape [batch_size]. The module tokenizes each string by splitting on spaces.
        default a fixed mean-pooling of all contextualized word representations with shape

        """
        if self.token:
            tokens  = x[0]
            seq_len = K.squeeze(x[1], axis=1)
            inputs = {'tokens':tokens, 'sequence_len':seq_len}
            result  = self.elmo(inputs=inputs, signature='tokens', as_dict=True)['elmo']
        else:
            result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1), signature='default', as_dict=True)['default']

        return result

    def compute_output_shape(self, input_shape):
        if self.token:
            return input_shape[0][0], input_shape[0][1],  self.dimension
        else:
            return input_shape[0], self.dimension


class Mention_Embedding(object):

    def __init__(self, filters=120, embed_size=300):
        self.filters = filters
        self.embed_size = embed_size

    def build(self):
        P_Fea1 = Input(shape=(3, self.embed_size))  # Embedding of Parents, Mention, and Suceeding Word: String Features
        P_Fea2 = Input(shape=(6, self.embed_size))  # Embeddings of 3 proceedings words, 3 succedings words of m
        P_Fea3 = Input(shape=(3, self.embed_size))

        Antecedent_Fea1 = Input(shape=(3, self.embed_size))
        Antecedent_Fea2 = Input(shape=(6, self.embed_size))
        Antecedent_Fea3 = Input(shape=(3, self.embed_size))
        Dist_Fea = Input(shape=(2,))

        Dist_Embed = Dense(self.filters, use_bias=True)(Dist_Fea)

        Mention_Represent1 = self.mention_embed(P_Fea1, P_Fea2, P_Fea3, 'Mention')
        Mention_Represent2 = self.mention_embed(Antecedent_Fea1, Antecedent_Fea2, Antecedent_Fea3, 'Antecedent')

        x = self.mentionpair_embed(Mention_Represent1, Mention_Represent2)
        x = Concatenate(name='Mention_Pair_Embedding')([x, Dist_Embed])

        model = Model([P_Fea1, P_Fea2, P_Fea3, Antecedent_Fea1, Antecedent_Fea2, Antecedent_Fea3, Dist_Fea], x)

        return model

    def mention_embed(self, inp1, inp2, inp3, target):
        Conv1_fea1 = self.Conv1k(inp1, [1, 2, 3])  # n_gram
        Conv1_fea2 = self.Conv1k(inp2, [1, 2, 3])  # n_gram
        Conv1_fea3 = self.Conv1k(inp3, [1, 2, 3])

        self.Expand_dim = Lambda(lambda x: K.expand_dims(x, axis=1))
        Conv1_fea1 = self.Expand_dim(Conv1_fea1)
        Conv1_fea2 = self.Expand_dim(Conv1_fea2)
        Conv1_fea3 = self.Expand_dim(Conv1_fea3)
        Conv2_Input = Concatenate(axis=1)([Conv1_fea1, Conv1_fea2, Conv1_fea3])

        x = Conv2D(self.filters, kernel_size=(3, 3), activation='tanh')(Conv2_Input)
        x = MaxPool2D(pool_size=(1, 1))(x)
        x = Lambda(lambda x: K.squeeze(x, axis=1), name="{}_Embed".format(target))(x)

        return x

    def Conv1k(self, x, kernels):
        assert len(kernels) != 0
        convs = []
        shape = x.get_shape().as_list()
        for kernel in kernels:
            conv = Conv1D(self.filters, kernel, activation='tanh')(x)
            pool = MaxPool1D(pool_size=int(shape[1] - kernel + 1))(conv)
            pool = Dropout(0.8)(pool)
            convs.append(pool)

        convs = Concatenate(axis=1)(convs)

        return convs

    def mentionpair_embed(self, M1, M2):
        x = Concatenate(axis=1)([M1, M2])
        x = Conv1D(self.filters, kernel_size=2, activation='tanh')(x)
        x = MaxPool1D(pool_size=1)(x)
        x = Dropout(0.8)(x)
        x = Flatten()(x)

        return x


class Coreference_Classifier(object):

    def __init__(self, Mention_Pair, Mention_Embedding, filters=60, embed_size=300):
        self.filters = filters
        self.embed_size = embed_size
        self.Mention_Pair = Mention_Pair
        self.Mention_Embed = Mention_Embedding

    def build(self):
        M1 = Input(shape=(3, self.embed_size))  # Embedding of Parents, Mention, and Suceeding Word: String Features
        M2 = Input(shape=(6, self.embed_size))  # Embeddings of 3 proceedings words, 3 succedings words of m
        M3 = Input(shape=(3, self.embed_size))  # Average Embedding of 3 proceeding sentence, 1 succeding sentence, and current sentence
        A1 = Input(shape=(3, self.embed_size))
        A2 = Input(shape=(6, self.embed_size))
        A3 = Input(shape=(3, self.embed_size))
        B1 = Input(shape=(3, self.embed_size))
        B2 = Input(shape=(6, self.embed_size))
        B3 = Input(shape=(3, self.embed_size))
        Dist_M_A = Input(shape=(2,))  # Mention and Antecedent A
        Dist_M_B = Input(shape=(2,))  # Mention and Antecedent B

        self.Expand_dim = Lambda(lambda x: K.expand_dims(x, axis=1))
        Mention_Pair1 = self.Mention_Pair([M1, M2, M3, A1, A2, A3, Dist_M_A])
        Mention_Pair2 = self.Mention_Pair([M1, M2, M3, B1, B2, B3, Dist_M_B])
        Mention_embedding = self.Mention_Embed([M1, M2, M3])
        Mention_Embedding = Flatten()(Mention_embedding)

        output1 = Concatenate()([Mention_Pair1, Mention_Pair2, Mention_Embedding])
        output1 = BatchNormalization()(output1)
        output1 = Dense(self.filters, use_bias=True, activation='relu')(output1)
        output1 = Dense(self.filters, use_bias=True, activation='relu')(output1)
        output1 = Dense(3, use_bias=True, activation='softmax', name='cluster_output')(output1)

        pair1 = self.cluster_classifier(Mention_Pair1, "pair1")
        pair2 = self.cluster_classifier(Mention_Pair2, "pair2")
        output2 = Add(name='singleton_output')([pair1, pair2])

        model = Model([M1, M2, M3, A1, A2, A3, B1, B2, B3, Dist_M_A, Dist_M_B], [output1, output2])

        return model

    def CONVs(self, x):
        x1 = GlobalAvgPool1D()(x)
        x2 = GlobalMaxPool1D()(x)
        x1 = self.Expand_dim(x1)
        x2 = self.Expand_dim(x2)
        x = Concatenate(axis=1)([x1, x2])
        x = Conv1D(self.filters, kernel_size=2)(x)
        x = MaxPool1D(pool_size=1)(x)
        x = Flatten()(x)

        return x

    def CONVp(self, x):
        x = self.Expand_dim(x)
        x1 = GlobalAvgPool1D()(x)
        x2 = GlobalMaxPool1D()(x)
        x1 = self.Expand_dim(x1)
        x2 = self.Expand_dim(x2)
        x = Concatenate(axis=1)([x1, x2])
        x = Conv1D(self.filters, kernel_size=2)(x)
        x = MaxPool1D(pool_size=1)(x)
        x = Flatten()(x)

        return x

    def cluster_classifier(self, x, _name):
        x = Dense(self.filters, use_bias=True, activation='relu')(x)
        x = Dense(1, use_bias=True, activation='sigmoid', name='{}_output'.format(_name))(x)

        return x

Embedding_model = Mention_Embedding().build()
layer_name = 'Mention_Embed'
Mention_Pair = Model(Embedding_model.inputs, Embedding_model.output)
Mention_Embedding = Model([Embedding_model.inputs[0], Embedding_model.inputs[1], Embedding_model.inputs[2]],
                          Embedding_model.get_layer(layer_name).output)

model = Coreference_Classifier(Mention_Pair, Mention_Embedding).build()
model.summary()
model.compile(optimizer='adam',
              loss={'cluster_output':'sparse_categorical_crossentropy', 'singleton_output':'mse'},
              loss_weights={'cluster_output': 1.0, 'singleton_output': 1.0},
              metrics={'cluster_output':"sparse_categorical_accuracy"})