from keras.layers import *
from keras.models import *
import keras.backend as K


class MentionPairEmbeding(object):

    def __init__(self, filters=280, max_len=50, embed_size=50, max_features=80000):
        self.filters = filters
        self.embed_size = embed_size
        self.max_len = max_len
        self.max_features = max_features

    def build(self):

        M1 = Input(shape=(3, self.embed_size))
        M2 = Input(shape=(6, self.embed_size))  # Embeddings of 3 proceedings words, 3 succedings words of m
        M3 = Input(shape=(5, self.max_len * self.embed_size))  # 3 proceeding sentence, 1 succeding sentence, current sentence
        A1 = Input(shape=(3, self.embed_size))
        A2 = Input(shape=(6, self.embed_size))
        A3 = Input(shape=(5, self.max_len * self.embed_size))
        B1 = Input(shape=(3, self.embed_size))
        B2 = Input(shape=(6, self.embed_size))
        B3 = Input(shape=(5, self.max_len * self.embed_size))

        Mention_embedding = self.mention_embed(M1, M2, M3)
        A_embedding = self.mention_embed(A1, A2, A3)
        B_embedding = self.mention_embed(B1, B2, B3)

        Mention_Pair1 = self.mentionpair_embed(Mention_embedding, A_embedding)
        Mention_Pair2 = self.mentionpair_embed(Mention_embedding, B_embedding)
        # Score_Layer
        Linear = Dense(1, use_bias=True)
        Mention_embedding = Flatten()(Mention_embedding)
        ScoreM = Linear(Mention_embedding)
        ScoreA = Linear(Mention_Pair1)
        ScoreB = Linear(Mention_Pair2)

        output = Concatenate()([ScoreA, ScoreB, ScoreM])
        output = Activation('softmax')(output)

        model = Model([M1, M2, M3, A1, A2, A3, B1, B2, B3], output)

        return model

    def mention_embed(self, inp1, inp2, inp3):

        Conv1_fea1 = self.Conv1k(inp1, [1, 2, 3])
        Conv1_fea2 = self.Conv1k(inp2, [1, 2, 3])
        Conv1_fea3 = self.Conv1k(inp3, [1, 2, 3])

        self.Expand_dim = Lambda(lambda x: K.expand_dims(x, axis=1))
        Conv1_fea1 = self.Expand_dim(Conv1_fea1)
        Conv1_fea2 = self.Expand_dim(Conv1_fea2)
        Conv1_fea3 = self.Expand_dim(Conv1_fea3)
        Conv2_Input = Concatenate(axis=1)([Conv1_fea1, Conv1_fea2, Conv1_fea3])

        x = Conv2D(self.filters, kernel_size=(3, 3))(Conv2_Input)
        x = Lambda(lambda x: K.squeeze(x, axis=1))(x)

        return x

    def mentionpair_embed(self, M1, M2):
        x = Concatenate(axis=1)([M1, M2])
        x = Conv1D(self.filters, kernel_size=2)(x)
        x = MaxPool1D(pool_size=1)(x)
        x = Flatten()(x)
        x = Dense(self.filters, use_bias=True, activation='relu')(x)

        return x

    def Conv1k(self, x, kernels):
        assert len(kernels) != 0
        convs = []
        shape = x.get_shape().as_list()
        for kernel in kernels:
            conv = Conv1D(self.filters, kernel)(x)
            pool = MaxPool1D(pool_size=int(shape[1] - kernel + 1))(conv)
            convs.append(pool)

        convs = Concatenate(axis=1)(convs)

        return convs