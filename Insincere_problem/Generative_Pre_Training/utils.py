from keras import backend as K
from keras.engine.topology import Layer


class Position_Embedding(Layer):

    def __init__(self, size=None, **kwargs):

       """
       "positional encodings" to the input embeddings at the
       bottoms of the encoder and decoder stacks.

       """
       self.size = size
       super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):

        if self.size == None:
            self.size = int(x.shape[-1]) #embed_size
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2*K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1) -1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.sin(position_ij), K.cos(position_ij)], axis=2)
        position_embeding = position_ij + x

        return position_embeding #batch_size, max_len, embed_size

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):

    """
    Attention(Q, K, V) = softmax(Q*KT/√dk)*V

    MultiHead(Q, K, V) = Concat(head1, ..., headh)
          where headi = Attention(QW*Qi, KW*Ki, V*WVi)
    """
    def __init__(self, n_head, output_dim=768, **kwargs):

        """
        h = 8 parallel attention layers

        """
        self.n_head= n_head
        self.output_dim = output_dim
        self.per_head_dim = int(output_dim / n_head)
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        e projections are parameter matrices
        WQi ∈ Rdmodel×dk ,
        WKi ∈ Rdmodel×dk ,
        WVi ∈ Rdmodel×dv and WO ∈ R hdv×dmodel
        """

        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        super(MultiHeadAttention, self).build(input_shape)



    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):

        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.n_head, self.per_head_dim))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.n_head, self.per_head_dim))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.n_head, self.per_head_dim))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #batch_dot is used to compute dot product of x and y when
        #x and y are data in batches, i.e. in a shape of (batch_size, :).
        #axes is the target_dim to be reduce
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.per_head_dim**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')

        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class LayerNormalization(Layer):

    def __init__(self, eps=1e-6, **kwargs):

        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma  = self.add_weight(name='gamma',
                                     shape=(input_shape[-1:]),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.beta = self.add_weight(name='bata',
                                  shape=(input_shape[-1:]),
                                  initializer='glorot_uniform',
                                  trainable=True)

        super(LayerNormalization,self).build(input_shape)

    def call(self, x):
        means = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        x_norm = (x - means) / ((std + self.eps)**0.5)
        output = self.gamma * x_norm + self.beta

        return output

    def compute_output_shape(self, input_shape):
        return input_shape
