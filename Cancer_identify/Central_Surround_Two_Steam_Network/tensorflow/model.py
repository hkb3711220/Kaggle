import tensorflow  as tf

class model(object):

    def __init__(self, image_shape, n_pair=2, is_training=True):

        #########################
        ##Change to 2 channel
        #########################
        self.image_shape = image_shape
        assert len(self.image_shape) == 3
        self.n_pair = n_pair

        self.h, self.w, self.c = self.image_shape[0], self.image_shape[1], self.image_shape[2]
        self.central_pair = tf.placeholder(shape=(None, self.n_pair, self.h, self.w, self.c), dtype=tf.float32)
        self.surrond_pair = tf.placeholder(shape=(None, self.n_pair, self.h, self.w, self.c), dtype=tf.float32)
        self.y            = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.is_training  = is_training

    def branch(self, inputs, scope):
        """
        based on ResNet34+SPP
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE,
                               initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(0.001)):

            inp = tf.pad(inputs, [[0,0],[2,2],[2,2],[0,0]])
            net = tf.layers.conv2d(inputs=inp, filters=64, kernel_size=[5,5], strides=(2,2),
                                     padding='same', activation=tf.nn.relu, name='conv_1', use_bias=True)
            net = tf.layers.max_pooling2d(net, strides=2, pool_size=[2,2], padding='same')
            net = self.block(net, 64, 3, scope='conv2')
            net = self.block(net, 128, 4, scope='conv3')
            net = self.block(net, 256, 6, scope='conv4')
            net = self.block(net, 512, 3, scope='conv5')
            #net = self.spp(net, pyramid_level=[1,2], scope='spp_layer')
            net = tf.keras.layers.GlobalAvgPool2D()(net)

        return net


    def block(self, inputs, n_out, n_block, scope):

        with tf.variable_scope(scope):
            out = self.blockneck(inputs, n_out, scope='blockneck1')
            for i in range(1, n_block):
                out = self.blockneck(out, n_out, scope='blockneck%s'%(i+1))

        return out

    def blockneck(self, inputs, n_out, scope, strides=(1,1)):

        n_channel = inputs.get_shape()[-1]
        if n_out != n_channel: strides=(2,2)

        with tf.variable_scope(scope):
            #Stride=2 -> downSampling
            #Pre-Activation
            h = tf.layers.batch_normalization(inputs, trainable=self.is_training, name='bn_1')
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(inputs=h, filters=n_out, kernel_size=[3,3], strides=strides,
                                   padding='same', name='bottleneck1_conv1')
            h = tf.layers.batch_normalization(h, trainable=self.is_training, name="bn_2")
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(inputs=h, filters=n_out, kernel_size=[3,3], strides=(1,1),
                                 padding='same', name='bottleneck1_conv2')

            if n_out != n_channel:
                shortcut = tf.layers.batch_normalization(inputs, trainable=self.is_training, name='bn_3')
                shortcut = tf.layers.conv2d(inputs=shortcut, filters=n_out, kernel_size=[3,3], strides=strides, padding='same')
            else:
                shortcut = inputs

        return shortcut+h

    def spp(self, inputs, pyramid_level, scope):

        pools = []
        h, w = inputs.get_shape().as_list()[1], inputs.get_shape().as_list()[2]
        n_channel = inputs.get_shape()[-1]

        with tf.variable_scope(scope):
            for l in pyramid_level:
                strides = int(h/l)
                pool_size = [strides, strides]
                pool = tf.layers.max_pooling2d(inputs, strides=strides, pool_size=pool_size, padding='same')
                pool = tf.layers.flatten(pool)
                pools.append(pool)

        return tf.concat(pools, axis=1)

    def cosine_similarity(self, l, r):

        l_norm = tf.nn.l2_normalize(l, axis=1)
        r_norm = tf.nn.l2_normalize(r, axis=1)

        similarity = l_norm * r_norm

        return similarity

    def l2_Euclidean_distance(self, f1, f2):

        p1 = f1[0]
        p2 = f2[0]
        q1 = f1[1]
        q2 = f2[1]

        dis = tf.sqrt(tf.square((q1 - p1)) + tf.square((q2-p2)))

        return dis

    def build(self):

        f_set1 = [self.branch(c, scope='branch_1') for c in tf.unstack(self.central_pair, axis=1)]
        f_set2 = [self.branch(s, scope='branch_2') for s in tf.unstack(self.surrond_pair, axis=1)]

        dis = self.l2_Euclidean_distance(f_set1, f_set2)

        out = tf.layers.dense(dis, 1, use_bias=True)
        pred = tf.nn.sigmoid(out)

        return out, pred

out, pred = model(image_shape=(96, 96, 3)).build()
