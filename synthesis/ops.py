import tensorflow as tf



def group_norm(input, G=8, name="group_norm"):

    with tf.variable_scope(name):

        N,W,H,D,C = input.get_shape()
        input_new = tf.transpose(input, perm=[0,4,1,2,3])
        input_new = tf.reshape(input_new, [N,G, C//G,W,H,D])
        
        mean, variance = tf.nn.moments(input_new, axes=[2,3,4,5], keep_dims=True)
        epsilon = 1e-5
        input_new = (input_new - mean) / tf.sqrt(variance + epsilon)
        
        gamma = tf.get_variable("GroupNorm_gamma", [C], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        beta = tf.get_variable("GroupNorm_beta", [C], initializer=tf.constant_initializer(0.0))
        
        gamma = tf.reshape(gamma, [1,C,1,1,1])
        beta = tf.reshape(beta, [1,C,1,1,1])
        
        gn_output = tf.reshape(input_new, [N,C,W,H,D]) * gamma + beta
        gn_output = tf.transpose(gn_output, perm=[0,2,3,4,1])
        
        return gn_output   


def condition_instance_norm(input, scale, offset, name="condition_instance_norm"):

    with tf.variable_scope(name):
        
        N,W,H,D,C = input.get_shape()
        input_new = tf.transpose(input, perm=[1,2,3,0,4])
        input_new = tf.reshape(input_new, [1,W,H,D,N*C])
        
        mean, variance = tf.nn.moments(input_new, axes=[1,2,3], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        
        normalized = tf.reshape((input_new-mean)*inv, [W,H,D,N,C])
        normalized = tf.transpose(normalized, perm=[3,0,1,2,4])
        
        return scale*normalized + offset


# follow Robust-Mseg.
def conv3d(input_, output_dim, ks=3, s=1, paddings='same', stddev = 0.01, name="conv3d"):

    with tf.variable_scope(name):

        conv = tf.layers.conv3d(
            input_, output_dim, ks, 
            strides=s,
            padding=paddings,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer=None)

        return conv


# defined by myself.
def deconv3d(input_, output_dim, ks=3, s=2, stddev = 0.01, name="conv3d"):
    
    with tf.variable_scope(name):
        
        deconv = tf.layers.conv3d_transpose(
            input_, output_dim, ks, 
            strides=s,
            padding='SAME', 
            activation=None,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer=None)
        
        return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def dense1d(input_, output_dim, name="dense1d"):
    with tf.variable_scope(name):
        return tf.layers.dense(input_, output_dim, activation=tf.nn.tanh, use_bias=True, trainable=True)
        


def flip_gradient(input_, l=1.0, name='flip_gradient'):
    with tf.variable_scope(name):
        positive_path = tf.stop_gradient(input_ * tf.cast(l + 1, tf.float32))
        negative_path = - input_ * tf.cast(l, tf.float32)
        return positive_path + negative_path