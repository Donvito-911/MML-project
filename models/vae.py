import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

from models._custom_layers import *

class VAE(Model):
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
        imgs = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(imgs)
            reconstruction = self.decoder(z)
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
    
    def call(self, input_image):
        _, _, z = self.encoder(input_image)
        return self.decoder(z)

class CVAE(VAE):
    
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
    
    def call(self, data):
        img, label = data
        _, _, z = self.encoder([img, label])
        return self.decoder([z, label])
    

def create_VAE(shape_img, latent_dim, n_blocks=19, n_cat=None, conditional=False):
    if (conditional and n_cat is None) or (not conditional and n_cat is not None):
        raise Exception("Bad setup. check n_cat and conditional arguments")
        
    max_neurons=2048
    if n_blocks == 19:
        max_neurons = 4096
    encoder = create_encoder(shape_img, latent_dim, n_blocks, max_neurons, n_cat=n_cat, 
                             conditional=conditional)
    decoder = create_decoder(latent_dim, n_blocks, max_neurons, n_cat=n_cat, 
                             conditional=conditional)
    vae = CVAE(encoder, decoder) if conditional else VAE(encoder, decoder)
    return vae

## ENCODER
def create_encoder(shape_img, latent_dim, n_blocks, max_neurons, n_cat=None, conditional=False):
    enc_input_image = Input(shape=(shape_img), name="input_image")
    if conditional:
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
    if conditional:
        flattened = Concatenate()([flattened, enc_input_label])
    enc_hidden = Dense(max_neurons)(flattened)
    enc_hidden = Dropout(0.5)(enc_hidden)
    enc_hidden = Dense(max_neurons//2)(enc_hidden)
    enc_hidden = Dropout(0.3)(enc_hidden)
    enc_hidden = Dense(max_neurons//4)(enc_hidden)
    z_mean = Dense(latent_dim, activation='linear', name="mu")(enc_hidden)
    z_log_var = Dense(latent_dim, activation='linear', name="l_sigma")(enc_hidden)
    z = Sampling()([z_mean, z_log_var])
    
    if conditional:
        encoder = Model(inputs=[enc_input_image, enc_input_label], outputs=[z_mean, z_log_var, z], name="encoder")
    else:
        encoder = Model(inputs=[enc_input_image], outputs=[z_mean, z_log_var, z], name="encoder")
    return encoder

# DECODER

def create_decoder(latent_dim, n_blocks, max_neurons, n_cat=None, conditional=False):
    dec_inp_latent_vector = Input(shape=(latent_dim), name="input_latent_vector")
    if conditional:
        decoder_inp_label = Input(shape=(n_cat), name = "input_label")
        dec_inp = Concatenate(name="decoder_concat")([dec_inp_latent_vector, decoder_inp_label])
    else:
        dec_inp = dec_inp_latent_vector
        
    unassigned = n_blocks-13
    
    dec_hidden = Dense(max_neurons//4, activation="relu")(dec_inp)
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

    if conditional:
        decoder = Model(inputs=[dec_inp_latent_vector, decoder_inp_label], outputs=dec_output_img, name="decoder")
    else:  
        decoder = Model(inputs=dec_inp_latent_vector, outputs=dec_output_img, name="decoder")
    return decoder

