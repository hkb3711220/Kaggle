from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.keras.backend import batch_dot

class ConvSelfAttention(Layer):

    def __init__(self, kernel_size, n_head, output_dim, use_bias=True, V_activation='elu', **kwargs):

        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.n_head = n_head
        self.use_bias = use_bias
        self.V_activation = V_activation
        self.per_head_dim = int(self.output_dim / self.n_head)
        super(ConvSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.WQ = self.add_weight(name='WQ',
                                  shape=(self.kernel_size, input_shape[-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(self.kernel_size, input_shape[-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(self.kernel_size, input_shape[-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        if self.use_bias:
            self.BQ = self.add_weight(name='BQ',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)
            self.BK = self.add_weight(name='BK',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)
            self.BV = self.add_weight(name='BV',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)

        super(ConvSelfAttention, self).build(input_shape)

    def call(self, x):

        Q_seq = x
        K_seq = x
        V_seq = x

        Q_seq = self.feature_extraction(Q_seq, self.WQ, self.BQ, activation='elu')
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.n_head, self.per_head_dim)) #(batch_size, max_len, n_head, per_head_dim)
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = self.feature_extraction(K_seq, self.WK, self.BK, activation='elu')
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.n_head, self.per_head_dim))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = self.feature_extraction(V_seq, self.WV, self.BV, activation=self.V_activation)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.n_head, self.per_head_dim))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = batch_dot(Q_seq, K_seq, axes=[3,3]) / self.per_head_dim**0.5
        A = K.softmax(A)
        A = batch_dot(A, V_seq, axes=[3,2])
        A = K.permute_dimensions(A, (0,2,1,3))
        A = K.reshape(A, (-1, K.shape(A)[1], self.output_dim))

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
        return (input_shape[0], input_shape[1]-self.kernel_size+1, input_shape[2])
