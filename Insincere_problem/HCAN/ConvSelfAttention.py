from keras import backend as k
from keras.engine.topology import Layer

class ConvSelfAttention(Layer):

    def __init__(self, n_head, output_dim, **kwargs):

        super(ConvSelfAttention, self).__init__(**kwargs)
        self.n_head     = n_head
        self.output_dim = output_dim

    def call(self, inputs):

        assert len(inputs) == 3
        Q, K, V = inputs
        Q       = k.reshape(Q, (-1, k.shape(Q)[1], self.n_head, int(self.output_dim /self.n_head)))
        K       = k.reshape(Q, (-1, k.shape(K)[1], self.n_head, int(self.output_dim /self.n_head)))
        V       = k.reshape(Q, (-1, k.shape(V)[1], self.n_head, int(self.output_dim /self.n_head)))

        heads = []
        for i in range(self.n_head):
            Qi = Q[:,:,i,:]
            Ki = K[:,:,i,:]
            Vi = V[:,:,i,:]

            a = k.batch_dot(Qi, Ki, axes=[2, 2]) / ((self.output_dim / self.n_head) ** 0.5)
            a = k.softmax(a)
            a = k.batch_dot(a, Vi, axes=[2, 1])
            heads.append(a)

        heads   = k.concatenate(heads)

        return heads
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2]
