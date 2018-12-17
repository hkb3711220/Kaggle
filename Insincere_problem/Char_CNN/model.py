import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.layers import Input, Conv1D, Embedding, MaxPool1D, Dense, Dropout, Flatten, Concatenate
from keras.optimizers import adam
from keras.models import Model
from keras.initializers import RandomNormal

class char_cnn(object):

    def __init__(self, num_filter, max_features=70, max_len=100, embed_size=69, pretrain_embeding_matrix=None):

        self.max_features = max_features
        self.max_len = max_len
        self.embed_size = embed_size
        self.num_filter = num_filter
        self.num_output = num_output
        self.pretrain_embed_matrix = pretrain_embeding_matrix
        self.adam = adam(lr=0.01, decay=1e-6)
        self.initializers = RandomNormal(mean=0.0, stddev=0.05)

    def get(self):

        inputs = Input(shape=(self.max_len, self.max_word_len,))
        x = Embedding(self.max_features, self.embed_size,)(inputs) # shape: batch_size, max_len, emb_size


        #dropout on the penultimate layer with a constraint on l2-norms of the weight vectors(Hintonetal., 2012).

        return model

model = char_cnn(num_filter=42, num_output=96, pretrain_embeding_matrix=embedding_weights).get()
model.summary()
