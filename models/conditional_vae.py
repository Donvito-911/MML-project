import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

from model._custom_layers import Sampling, conv_block, conv_block_transpose

class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        imgs, labels = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([imgs, labels])
            reconstruction = self.decoder([z, labels])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(imgs, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def create_conditional_VAE(shape_img_input, n_cat, latent_dim):
    encoder = create_encoder(shape_img_input, n_cat, latent_dim)
    decoder = create_decoder(latent_dim, n_cat)
    conditional_vae = CVAE(encoder, decoder)
    return conditional_vae


def create_encoder(shape_img, n_cat, latent_dim, n_blocks, max_neurons):
    enc_input_image = Input(shape=(shape_img), name="input_image")
    enc_input_label = Input(shape=(n_cat), name="input_label")
    unassigned = max(0, n_blocks-13)
    # convolutionals block
    enc_conv = conv_block(enc_input_image, filters=64, use_bn=True) # 128x128x64
    enc_conv = conv_block(enc_conv, filters=64, strides=(2,2), use_dropout=True) # 64x64x64

    enc_conv = conv_block(enc_conv, filters=128) # 64x64x128
    enc_conv = conv_block(enc_conv, filters=128, strides=(2,2), use_bn=True, use_dropout=True) # 32x32x128

    for filters in [256, 512]:
        assigning = min(3, unassigned)
        unassigned -= assigning
        while assigning > 0:
            enc_conv = conv_block(enc_conv, filters=filters, use_bn=True, use_dropout=True)
            assigning-=1
        enc_conv = conv_block(enc_conv, strides=(2,2), filters=filters, use_bn=True, use_dropout=True)
    
    enc_conv = conv_block(enc_conv, filters=512) # 8x8x512
    enc_conv = conv_block(enc_conv, filters=512, use_dropout=True) # 8x8x512
    enc_conv = conv_block(enc_conv, filters=512) # 8x8x512
    enc_conv = conv_block(enc_conv, filters=512, strides=(2,2), use_bn=True, use_dropout=True) # 4x4x512


    # concat
    flattened = Flatten()(enc_conv)
    enc_concat = Concatenate()([flattened, enc_input_label])


    enc_hidden = Dense(max_neurons)(enc_concat)
    enc_hidden = Dropout(0.5)(enc_hidden)
    enc_hidden = Dense(max_neurons//2)(enc_hidden)
    enc_hidden = Dropout(0.3)(enc_hidden)
    enc_hidden = Dense(max_neurons//4)(enc_hidden)
    z_mean = Dense(latent_dim, activation='linear', name="mu")(enc_hidden)
    z_log_var = Dense(latent_dim, activation='linear', name="l_sigma")(enc_hidden)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model([enc_input_image, enc_input_label], [z_mean, z_log_var, z], name="encoder")
    return encoder

def create_decoder(latent_dim, n_cat, n_blocks, max_neurons):
    dec_inp_latent_vector = Input(shape=(latent_dim), name="input_latent_vector")
    decoder_inp_label = Input(shape=(n_cat), name = "input_label")
    dec_concat = Concatenate(name="decoder_concat")([dec_inp_latent_vector, decoder_inp_label])
    unassigned = n_blocks-13
    
    dec_hidden = Dense(max_neurons//4, activation="relu")(dec_inp_latent_vector)
    dec_hidden = Dropout(0.3)(dec_hidden)
    dec_hidden = Dense(max_neurons//2, activation="relu")(dec_hidden)
    dec_hidden = Dropout(0.3)(dec_hidden)
    dec_hidden = Dense(max_neurons, activation="relu")(dec_hidden)
    dec_hidden = Dense(4*4*512, activation="relu", name="hidden_layer7")(dec_hidden)
    dec_hidden = Dropout(0.3)(dec_hidden)
    reshaped = Reshape(target_shape=(4,4,512))(dec_hidden)

    dec_conv = conv_block_transpose(reshaped, filters=512, use_bn=True, use_dropout=True)  # 4x4x512
    dec_conv = conv_block_transpose(dec_conv, filters=512)  # 4x4x512
    dec_conv = conv_block_transpose(dec_conv, filters=512, use_dropout=True)  # 4x4x512
    dec_conv = conv_block_transpose(dec_conv, filters=512, strides=(2,2), use_bn=True)  # 8x8x512

    for filters in [256, 512]:
        assigning = min(3, unassigned)
        unassigned -= assigning
        while assigning > 0:
            dec_conv = conv_block_transpose(dec_conv, filters=filters, use_bn=True, use_dropout=True)
            assigning-=1
        dec_conv = conv_block_transpose(dec_conv, strides=(2,2), filters=filters, use_dropout=True)


    dec_conv = conv_block_transpose(dec_conv, filters=128, use_bn=True, use_dropout=True)  # 32x32x128
    dec_conv = conv_block_transpose(dec_conv, filters=128, strides=(2,2), use_bn=True)  # 64x64x128

    dec_conv = conv_block_transpose(dec_conv, filters=64, use_bn=True, use_dropout=True)  # 64x64x64
    dec_conv = conv_block_transpose(dec_conv, filters=128, strides=(2,2), use_bn=True)  # 128x128x64

    dec_output_img = Conv2DTranspose(filters=1, kernel_size=3, padding="same", activation="sigmoid")(dec_conv)


    decoder = Model([dec_inp_latent_vector, decoder_inp_label], dec_output_img, name="decoder")
    return decoder
