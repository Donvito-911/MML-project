"""Hparam explorer module"""
from model_builder import WCGANGP
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


class HparamExplorer:

    def __init__(
        self,
        train_file,
        latent_dims,
        batch_size,
        n_epochs,
        generators,
        discriminators
    ):
        with h5py.File(train_file,'r+') as f:
            self.train_images = (np.array(f['X_train']).astype("float32")) / 255
            self.train_labels = np.array(f['y_train']).astype("float32")

        self.batch_size = batch_size
        self.latent_dims = latent_dims
        self.n_epochs = n_epochs
        self.generators = generators
        self.discriminators = discriminators
          
    def train_single_model(self, generator, discriminator, latent_dim):

        def W_disc_loss(real_pred, fake_pred, eps, grad_pen):
            return -tf.reduce_mean(real_pred) + tf.reduce_mean(fake_pred) + eps * grad_pen
        
        def W_gen_loss(fake_pred):
            return -tf.reduce_mean(fake_pred)

        #@title Make the dataset
        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((self.train_images, self.train_labels))
            .shuffle(self.train_images.shape[0])
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        num_class = self.train_labels.shape[1]
        gen_input_shape = (latent_dim + num_class, )
        disc_input_shape = (self.train_images.shape[1],
                    self.train_images.shape[2],
                    self.train_images.shape[3] + num_class,)

        Generator = generator(gen_input_shape)
        Discriminator = discriminator(disc_input_shape)

        disc_opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        gen_opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        model = WCGANGP(
            Discriminator,
            Generator,
            latent_dim,
            d_extra_steps=5,
            gp_weight=10.0,
            num_class=num_class,
            disc_input_shape=disc_input_shape
        )
        model_config = f"{generator.__name__}__{discriminator.__name__}__{latent_dim}"
        log_dir = f"logs/{model_config}"
        tensorboard_callback =  tf.keras.callbacks.TensorBoard(
            log_dir,
            histogram_freq=5,
            profile_batch = '2,3'
        )
        print(f"Fitting: {model_config}")
        model.compile(disc_opt, gen_opt, W_disc_loss, W_gen_loss)
    
        _ = model.fit(
            train_dataset,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=1,
            callbacks=[tensorboard_callback]
        
        )

        #@title Number o images
        n_images = 4 #@param {type:"integer"}

        plt.close()
        images_path = f"logs/{model_config}/images"

        if not os.path.exists(images_path):
            os.mkdir(images_path)

        for ix in range(n_images):
        
            fixed_seed = tf.random.normal([num_class, latent_dim])

            labels = tf.range(0, num_class)
            one_hot_labels = tf.one_hot(labels, num_class)
            noise_labels = tf.concat([fixed_seed, one_hot_labels], axis=1)
            predictions = Generator(noise_labels, training=False)
            
            plt.figure(figsize=((num_class+1)*5, 2*5))
            for i in range(predictions.shape[0]):    
                plt.subplot(1, num_class, i+1)
                plt.imshow(predictions[i, :, :, ])
                plt.axis('off')
            plt.savefig(f"{images_path}/example_{ix}.png", bbox_inches='tight')
        
        Generator.save(f'logs/{model_config}/model', save_format='tf')

        tf.keras.backend.clear_session()

    def run(self):
        for generator in self.generators:
            for discriminator in self.discriminators:
                for latent_dim in self.latent_dims:
                    self.train_single_model(generator, discriminator, latent_dim)
