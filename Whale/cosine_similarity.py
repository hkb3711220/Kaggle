import numpy as np
from keras.engine import Layer
from keras import backend as K

def cosine_similarity(x, y, eps=1e-8):
    """
    The genral
    """

    nx = x / np.sqrt(np.sum(x**2)+eps)
    ny = y / np.sqrt(np.sum(y**2)+eps)
    return np.dot(nx, ny)

class cosinesimilarityLayer(Layer):

    """
    The layer for Mathching Net
    """

    def __init__(self, eps=1e-8, **kwargs):

        self.eps = eps
        super(cosinesimilarityLayer).__init__(**kwargs)

    def call(self, x):

        similarity_list = []

        try:
            assert len(x) == 2, 'The input should be like [x0, x1]'
        except AssertionError as err:
            print(err)

        fc_embedding = x[0] #(batch_size, num_of_featrues)
        gc_embedding = x[1] #(n_class*k_shot, batch_size, num_of_featrues)

        fc_norm = K.l2_normalize(fc_embedding, axis=1)

        for g in gc_embedding:
            g_norm = K.l2_normalize(g, axis=1) #g (batch_size, num_of_featrues)
            cosine_similarity = K.batch_dot(fc_norm, g_norm, axes=1)
            similarity_list.append(cosine_similarity)

        return K.squeeze(K.stack(similarity_list, axis=1), axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[1][1], input_shape[1][0])
