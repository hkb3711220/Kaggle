from keras.engine import Layer
from keras import backend as K

class AttentionLayer(Layer):

    """
    Attention(Q, K, V) = softmax(Q*KT/√dk)*V

    MultiHead(Q, K, V) = Concat(head1, ..., headh)
          where headi = Attention(QW*Qi, KW*Ki, V*WVi)
    """
    def __init__(self, num_attention_heads, output_dim, **kwargs):

        self.num_attention_heads = num_attention_heads
        self.output_dim = output_dim #hidden size
        self.size_per_head = int(output_dim / num_attention_heads)
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        WQi ∈ Rdmodel×dk ,
        WKi ∈ Rdmodel×dk ,
        WVi ∈ Rdmodel×dv and WO ∈ R hdv×dmodel
        """

        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        super(AttentionLayer, self).build(input_shape)


    def Mask(self, inputs, seq_len, mode='mul'):

        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1) #指定した軸に沿って累積和を計算します．
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):

        """
        self-Attention
        """

        query_layer, key_layer, Value_layer = x, x, x
        Q_len,V_len = None,None

        query_layer = K.dot(query_layer, self.WQ) #(batch_size, sequence_length, output_dim)
        query_layer = K.reshape(query_layer, (-1, K.shape(query_layer)[1], self.num_attention_heads, self.size_per_head))
        query_layer = K.permute_dimensions(query_layer, (0,2,1,3))

        key_layer = K.dot(key_layer, self.WK)
        key_layer = K.reshape(key_layer, (-1, K.shape(key_layer)[1], self.num_attention_heads, self.size_per_head))
        key_layer = K.permute_dimensions(key_layer, (0,2,1,3))

        Value_layer = K.dot(Value_layer, self.WV)
        Value_layer = K.reshape(Value_layer, (-1, K.shape(Value_layer)[1], self.num_attention_heads, self.size_per_head))
        Value_layer = K.permute_dimensions(Value_layer, (0,2,1,3))
        #batch_dot is used to compute dot product of x and y when
        #x and y are data in batches, i.e. in a shape of (batch_size, :).
        #axes is the target_dim to be reduce

        A = K.batch_dot(query_layer, key_layer, axes=[3,3]) / float(self.size_per_head**0.5)
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, Value_layer, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))

        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
