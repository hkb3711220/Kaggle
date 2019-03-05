from keras.layers import *
import keras.backend as K
from attention import AttentionLayer
from keras.models import Model

def dot_product(x, kernel):

    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):

    def __init__(self, bias=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.bias = bias

    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                name='attention_w',
                                initializer='glorot_uniform')
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                    name='attention_b',
                                    initializer='zero')

        self.u = self.add_weight(shape=(input_shape[-1],),
                                name='attention_u',
                                initializer='glorot_uniform')

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x):

        uit = dot_product(x, self.w)

        if self.bias:
            uit += self.b #(batch_size, max_len, embed_size)
        uit = K.tanh(uit) #(batch_size, max_len, embed_size)

        ait = dot_product(uit, self.u)

        a = K.exp(ait)
        a /= K.cast(K.sum(a, axis=1, keepdims=True)+ K.epsilon(), K.floatx())

        a = K.expand_dims(ait)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class charembed(object):

    def __init__(self, filters, kernel, max_word_len, embed_size=30,  max_char_features=26, highway=True):

        self.filters           = filters
        self.kernel            = kernel
        self.max_char_features = max_char_features
        self.max_word_len      = max_word_len
        self.embed_size        = embed_size
        self.highway           = highway

    def build(self):

        fea_ms = []
        pools = []

        inp   = Input(shape=(self.max_word_len,))
        #Embedding
        x     = Embedding(self.max_char_features, self.embed_size, input_shape=(self.max_word_len,))(inp)

        for i in range(len(self.kernel)):
            fea_ms.append(Conv1D(filters=self.filters[i], kernel_size=(self.kernel[i]), activation='tanh',
                         name='conv_{}'.format(i))(x))
        #Max over time pooling layer
        for i in range(len(fea_ms)):
            pools.append(GlobalMaxPooling1D(name='Maxovertimepoolinglayer_{}'.format(i))(fea_ms[i]))

        #High way
        feature_vectors = Concatenate(axis=1)(pools)
        transform_gate  = Dense(sum(self.filters), activation='sigmoid',name='transform_gate',
                          use_bias=True)(feature_vectors)
        carry_gate      = Lambda(lambda x: 1-x, name='carry_gate')(transform_gate)

        #Output
        z = Dense(sum(self.filters),activation='relu')(feature_vectors)
        z = add([multiply([z, transform_gate]), multiply([carry_gate, feature_vectors])])

        model = Model(inputs=inp, outputs=z)

        return model

class end2ed_coref(object):

    def __init__(self, charembed, max_len=50, max_word_len=30, max_features=50000,
                 embed_size=300, hidden_unit=400, pretrain_embed_matrix=None):

        self.max_len               = max_len
        self.max_word_len          = max_word_len
        self.max_features          = max_features
        self.embed_size            = embed_size
        self.pretrain_embed_matrix = pretrain_embed_matrix
        self.hidden_unit           = hidden_unit
        self.charembed             = charembed

    def _build(self):

        # Assume that We have 3 spans for train
        inpa = Input((self.max_len,))
        inpb = Input((self.max_len,))
        inpc = Input((self.max_len,))
        inputs = [inpa, inpb, inpc]


        char_inpa  = Input((self.max_len, self.max_word_len))
        char_inpb  = Input((self.max_len, self.max_word_len))
        char_inpc  = Input((self.max_len, self.max_word_len))
        char_inputs = [char_inpa, char_inpb, char_inpc]

        #Embedding Layer
        if self.pretrain_embed_matrix:
            Word_embed = Embedding(self.max_features, self.embed_size, weights=[self.pretrain_embed_matrix])
        else:
            Word_embed = Embedding(self.max_features, self.embed_size)
        Char_embed = TimeDistributed(self.charembed, input_shape=(self.max_len, self.max_word_len))

        word_embeds = [Word_embed(inp) for inp in inputs]
        char_embeds = [Char_embed(inp) for inp in char_inputs]

        Concat = Concatenate(axis=2)
        xs     = [Concat([embed1, embed2]) for embed1, embed2 in zip(word_embeds, char_embeds)]

         # Use Independent of the Bidirectional LSTM

        _xs    = []
        Dropout = SpatialDropout1D(0.5)
        Bilstm  = Bidirectional(LSTM(self.hidden_unit//2, return_sequences=True)) # (batch_size, max_len, num_units)

        for i in range(len(xs)):
            _x = Dropout(xs[i])
            _x = Bilstm(_x)
            _xs.append(_x)

        xs = _xs


        model = Model([inpa, inpb, inpc, char_inpa, char_inpb, char_inpc], xs)

        return model


#Build Model
char_embed = charembed(filters=[50, 50, 50], kernel=[3, 4, 5], max_word_len=30, embed_size=8).build()
model = end2ed_coref(char_embed)._build()
model.summary()
