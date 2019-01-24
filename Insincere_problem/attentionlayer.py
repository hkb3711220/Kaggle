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
