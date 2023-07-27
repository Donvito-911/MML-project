import tensorflow as tf
from tensorflow.keras.layers import *

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
   
    
def conv_block(
    x,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation = LeakyReLU(0.3),
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.3):
    x = Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x

def conv_block_transpose(
    x,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation = LeakyReLU(0.3),
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.3):
    x = Conv2DTranspose(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x
