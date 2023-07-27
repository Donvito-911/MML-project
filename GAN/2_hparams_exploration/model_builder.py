import tensorflow as tf
from tensorflow.keras import Model

"""Model builder module that uses a discriminator and generator"""
class WCGANGP(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        d_extra_steps=3,
        gp_weight=10.0,
        num_class=4,
        disc_input_shape=[150,150,7]
    ):
        super(WCGANGP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_extra_steps = d_extra_steps
        self.gp_weight = gp_weight
        self.num_class = num_class
        self.disc_input_shape = disc_input_shape

    def compile(self, disc_opt, gen_opt, d_loss_fn, g_loss_fn):
        super(WCGANGP, self).compile()
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

            This loss is calculated on an interpolated image
            and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1])
        interpolated = real_images * alpha + fake_images * (1 - alpha)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp

    # Override the training step function of the Model class
    def train_step(self, data):

      images, labels = data

      batch_size = tf.shape(images)[0]

      one_hot_labels = labels

      image_one_hot_labels = tf.repeat(one_hot_labels,
                                 repeats=[self.disc_input_shape[0] * self.disc_input_shape[1]])

      image_one_hot_labels = tf.reshape(image_one_hot_labels,
                                        (-1, self.disc_input_shape[0], self.disc_input_shape[1], self.num_class))

      #---------Discriminator loop--------------


      for i in range(self.d_extra_steps):

        noise = tf.random.normal([batch_size, self.latent_dim])

        noise_labels = tf.concat([noise, one_hot_labels], axis=1)

        with tf.GradientTape() as disc_tape:
          
          G_output = self.generator(noise_labels, training=True)
          G_output = tf.concat([G_output, image_one_hot_labels], -1)
          images_and_labels = tf.concat([images, image_one_hot_labels], -1)
  
          D_fake_pred = self.discriminator(G_output, training=True)

          D_real_pred = self.discriminator(images_and_labels, training=True)

          gp = self.gradient_penalty(batch_size, images_and_labels, G_output)
          D_loss = self.d_loss_fn(D_real_pred, D_fake_pred, self.gp_weight, gp)

        D_grad = disc_tape.gradient(D_loss, self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(D_grad, self.discriminator.trainable_variables))
      
      #------Generator loop-------------

      noise = tf.random.normal([batch_size, self.latent_dim])

      noise_labels = tf.concat([noise, one_hot_labels], axis=1)

      with tf.GradientTape() as gen_tape:

        G_output = self.generator(noise_labels, training=True)
        G_output = tf.concat([G_output, image_one_hot_labels], -1)

        D_fake_pred = self.discriminator(G_output, training=True)

        G_loss = self.g_loss_fn(D_fake_pred)

      G_grad = gen_tape.gradient(G_loss, self.generator.trainable_variables)
      self.gen_opt.apply_gradients(zip(G_grad, self.generator.trainable_variables))

      return {'d_loss': D_loss, "g_loss": G_loss}

