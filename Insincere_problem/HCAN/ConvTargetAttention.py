from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.keras.backend import batch_dot

class ConvTargetAttention(Layer):

    def __init__(self, kernel_size, n_head, output_dim, use_bias=True, **kwargs):

        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.n_head = n_head
        self.use_bias = use_bias
        self.per_head_dim = int(self.output_dim / self.n_head)
        super(ConvTargetAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.WK = self.add_weight(name='WK',
                                  shape=(self.kernel_size, input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(self.kernel_size, input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        if self.use_bias:
            self.BK = self.add_weight(name='BK',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)
            self.BV = self.add_weight(name='BV',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)

        super(ConvTargetAttention, self).build(input_shape)

    def call(self, inputs):

        assert len(inputs) == 3
        T_seq, K_seq, V_seq = inputs # T:(batch_size, embed_size)
        print(T_seq)
        T_seq = K.reshape(T_seq, (-1, self.n_head, self.per_head_dim))

        K_seq = self.feature_extraction(K_seq, self.WK, self.BK, activation='elu')
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.n_head, self.per_head_dim))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))  #(batch_size, n_head, seq_len-kernel_size+1, self.per_head_dim)
        V_seq = self.feature_extraction(V_seq, self.WV, self.BV, activation='elu')
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.n_head, self.per_head_dim))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3)) #(batch_size, n_head, seq_len-kernel_size+1, self.per_head_dim)

        A = batch_dot(T_seq, K_seq, axes=[2, 3]) / self.per_head_dim **0.5
        A = K.softmax(A)
        A = batch_dot(A, V_seq, axes=[1, 2])
        A = K.reshape(A, (-1, self.output_dim))

        return A

    def feature_extraction(self, inp, kernel, bias, activation='elu'):

            x = K.conv1d(inp, kernel=kernel)
            if self.use_bias:
                x = K.bias_add(x, bias)
            if activation == 'elu':
                x = K.elu(x)
            elif activation == 'tanh':
                x = K.tanh(x)

            return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]
