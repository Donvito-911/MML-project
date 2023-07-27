"""Discriminator builder methods"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def generator1(gen_input_shape):
  """Uses conv2dtranspose and batch normalization"""
  model = tf.keras.Sequential(name='generator1')

  model.add(layers.Dense(5*5*256, input_shape=gen_input_shape))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape([5, 5, 256]))
  model.add(layers.Conv2DTranspose(128, 5, 6, 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, 5, 5, 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(3, 5, 1, 'same', activation='sigmoid'))

  return model

def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    up_size=(2, 2),
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def generator2(gen_input_shape):
    """Uses upsampling blocks and batch normalization"""
    noise = layers.Input(shape=gen_input_shape)
    x = layers.Dense(5 * 5 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((5, 5, 256))(x)
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        up_size=(3, 3),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        up_size=(5, 5),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, 3, layers.Activation("sigmoid"), strides=(1, 1), use_bias=False, use_bn=True
    )
    
    g_model = keras.models.Model(noise, x, name="generator2")
    return g_model
