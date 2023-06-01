import os
import ad.constants

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl

from typing import List, Tuple

from ad import utils
from ad import metrics
from ad import layers


class HeimelJVAE(tf.keras.Model):
    """JVAE modified, inspired by 'QCD or What?'"""

    def __init__(self, 
                 image_shape: tuple, 
                 continous_latent: int = 32, 
                 discrete_latent: int = 16,
                 temperature: float = 50., 
                 beta: float = 3e3, 
                 eps_kl: float = 1e-7,
                 name=None, 
                 **kwargs):

        self._base_model_initialized = True

        # parameter assignment
        self.continous_latent = continous_latent
        self.discrete_latent = discrete_latent  # gumbel-softmax
        self.beta = beta
        self.eps_kl = eps_kl  # KL divergence between gumbel-softmax distribution

        # models
        self.encoder = self.get_encoder(input_shape=image_shape, **kwargs.get('encoder', {}))
        self.sampling = layers.joint_sampling(temp=temperature, name='joint_sampling', **kwargs)
        self.decoder = self.get_decoder(**kwargs.get('decoder', {}))

        # metrics
        self.trackers = dict(
            loss=tf.keras.metrics.Mean(name='total_loss'),
            kl_cont_loss_tracker=tf.keras.metrics.Mean(name="kl_cont_loss"),
            kl_disc_loss_tracker=tf.keras.metrics.Mean(name="kl_disc_loss"),
            mse=ad.metrics.MSE(name='mse'),
            ssim=ad.metrics.SSIM(name='ssim'),
            true_energy=tf.keras.metrics.Mean(name='true_energy'),
            pred_energy=tf.keras.metrics.Mean(name='pred_energy'),
            grads_norm=tf.keras.metrics.Mean(name='gradients_norm'),
            weights_norm=tf.keras.metrics.Mean(name='weights_norm')
        )

        self.test_trackers = [k for k in self.trackers.keys() if '_norm' not in k]

        super().__init__()

    def call(self, x: tf.Tensor, training=None):
        q, mean, var = self.encoder(x, training=training)
        z = self.sampling([mean, var, q])

        return self.decoder(z, training=training)

    @tf.function
    def train_step(self, batch: tf.Tensor):
        batch = self.augment(batch)

        with tf.GradientTape() as tape:
            q, z_mean, z_log_var = self.encoder(batch, training=True)
            z = self.sampling([z_mean, z_log_var, q])
            reconstruction = self.decoder(z, training=True)

            # losses
            #######################
            #######   RECO   ######
            #######################

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(batch, reconstruction), axis=(1, 2)
                ) 
            )
            
            #######################
            ####### KLD-LOSS ######
            #######################

            ##### kl_continous #####
            kl_cont_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_cont_loss = tf.reduce_sum(kl_cont_loss, axis=1)
            kl_cont_loss = tf.reduce_mean(self.beta * kl_cont_loss, axis = 0)
            
            ##### kl_categorical #####
            q_p = tf.nn.softmax(q, axis=-1) # Convert the categorical codes into probabilities
            # Entropy of the logits
            h1 = q_p * tf.math.log(q_p + self.eps_kl)
            # Cross entropy with the categorical distribution
            h2 = q_p * tf.math.log(1. / self.discrete_latent + self.eps_kl)
            kl_disc_loss = tf.reduce_mean(tf.reduce_sum(h1- h2 , axis = 1 ) * self.beta, axis = 0)
            # kl_disc_loss = abs(kl_disc_loss)

            total_loss = reconstruction_loss + kl_cont_loss + kl_disc_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.update_trackers(
            loss=total_loss,
            mse=(reconstruction, batch),
            ssim=(reconstruction, batch),
            kl_cont_loss_tracker=kl_cont_loss,
            kl_disc_loss_tracker=kl_disc_loss,
            true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
            pred_energy=tf.reduce_sum(reconstruction, axis=[1, 2, 3]),
            grads_norm=utils.tf_global_norm(gradients),
            weights_norm=utils.tf_global_norm(self.trainable_variables)
        )

        return {k: metric.result() for k, metric in self.trackers.items()}

    @tf.function
    def test_step(self, batch: tf.Tensor):
        q, z_mean, z_log_var = self.encoder(batch, training=False)
        z = self.sampling([z_mean, z_log_var, q])
        reconstruction = self.decoder(z, training=False)

        # losses
        #######################
        #######   RECO   ######
        #######################

        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(batch, reconstruction), axis=(1, 2)
                ) 
        )
            
        #######################
        ####### KLD-LOSS ######
        #######################

        ##### kl_continous #####
        kl_cont_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_cont_loss = tf.reduce_sum(kl_cont_loss, axis=1)
        kl_cont_loss = tf.reduce_mean(self.beta * kl_cont_loss, axis = 0)
            
        ##### kl_categorical #####
        q_p = tf.nn.softmax(q, axis=-1) # Convert the categorical codes into probabilities
        # Entropy of the logits
        h1 = q_p * tf.math.log(q_p + self.eps_kl)
        # Cross entropy with the categorical distribution
        h2 = q_p * tf.math.log(1. / self.discrete_latent + self.eps_kl)
        kl_disc_loss = tf.reduce_mean(tf.reduce_sum(h1- h2 , axis = 1 ) * self.beta, axis = 0)
        # kl_disc_loss = abs(kl_disc_loss)

        total_loss = reconstruction_loss + kl_cont_loss + kl_disc_loss

        self.update_trackers(
            loss=total_loss,
            mse=(reconstruction, batch),
            ssim=(reconstruction, batch),
            kl_cont_loss_tracker=kl_cont_loss,
            kl_disc_loss_tracker=kl_disc_loss,
            true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
            pred_energy=tf.reduce_sum(reconstruction, axis=[1, 2, 3]),
            weights_norm=utils.tf_global_norm(self.trainable_variables)
        )

        return {k: metric.result() for k, metric in self.trackers.items()}

    def get_encoder(self, input_shape: tuple, filters: List[Tuple[int, int]],
                    units: List[int], **kwargs):
        """ Build the encoder"""

        image = tfkl.Input(shape=input_shape, name='image')
        x = image

        for i, (f1, f2) in enumerate(filters):
            # conv-1
            x = tfkl.Conv2D(filters=f1, kernel_size=3, activation=None, padding='same',
                            name=f'conv1-{i}', **kwargs)(x)
            x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU1-{i}')(x)

            # conv-2
            if f2 is not None:
                x = tfkl.Conv2D(filters=f2, kernel_size=3, activation=None, padding='same',
                                name=f'conv2-{i}', **kwargs)(x)
                x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU2-{i}')(x)

            # pool
            x = tfkl.AvgPool2D(name=f'avg_pool-{i}')(x)

        x = tfkl.Flatten(name='flatten')(x)

        for i, num_units in enumerate(units):
            x = tfkl.Dense(num_units, activation=None, **kwargs, name=f'dense-{i}')(x)
            x = tfkl.PReLU(name=f'PReLU-{i}')(x)

        q = tfkl.Dense(units=self.discrete_latent, name='z_categorical')(x)
        encoded_mean = tfkl.Dense(units=self.continous_latent, name='z_mean')(x)
        encoded_var = tfkl.Dense(units=self.continous_latent, name='z_var')(x)

        return tf.keras.Model(inputs=image, outputs=[q, encoded_mean, encoded_var], name='Encoder')

    def get_decoder(self, filters: List[Tuple[int, int]], units: List[int],
                    reshape_to: tuple, crop=(72, 58), bias=0.0, out_activation=tf.nn.softmax, **kwargs):
        """ Build the decoder """

        z = tfkl.Input(shape=self.continous_latent + self.discrete_latent, name='z')
        x = z

        for i, num_units in enumerate(units):
            x = tfkl.Dense(num_units, name=f'dense-{i}', **kwargs)(x)
            x = tfkl.PReLU(name=f'PReLU-{i}')(x)

        x = tfkl.Reshape(reshape_to, name='reshape')(x)

        for i, (f1, f2) in enumerate(filters):
            x = tfkl.Conv2DTranspose(filters=f1, kernel_size=3, padding='same', **kwargs,
                                     strides=2, name=f'conv_t1-{i}')(x)
            x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU1-{i}')(x)

            x = tfkl.Conv2D(filters=f1, kernel_size=3, padding='same', **kwargs,
                            name=f'conv_2-{i}')(x)
            x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU2-{i}')(x)

        x = tfkl.CenterCrop(*crop, name='crop')(x)
        reco = tfkl.Conv2D(filters=1, kernel_size=1, activation=out_activation, padding='same',
                           bias_initializer=tf.keras.initializers.Constant(bias),
                           **kwargs)(x)

        return tf.keras.Model(z, reco, name='Decoder')

    @staticmethod
    def mse_loss(true, pred):
        loss = tf.reduce_sum(tf.square(true - pred), axis=[1, 2, 3])
        return tf.reduce_mean(loss)
    
    def kl_continuous_loss( self, mean, var):
        kl_loss = -0.5 * tf.reduce_sum(
            1 + tf.math.log(var) - tf.square(mean) - var, axis=1
        )
        return tf.reduce_mean(kl_loss)
    
    def kl_discrete_loss(self, q):
        q = tf.clip_by_value(q, 1e-8, 1 - 1e-8)
        kl_loss = tf.reduce_sum(q * (tf.math.log(q) - tf.math.log(1.0 / self.discrete_latent + self.eps_kl)), axis=1)
        return tf.reduce_mean(kl_loss)

    @tf.function
    def augment(self, x: tf.Tensor):
        return tf.map_fn(fn=ad.aug.tf_augment, elems=x, parallel_iterations=16)

    def update_trackers(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.trackers:
                continue

            if isinstance(v, tuple):
                self.trackers[k].update_state(*v)

            elif isinstance(v, dict):
                self.trackers[k].update_state(**v)
            else:
                self.trackers[k].update_state(v)

    def summary(self, **kwargs):
        self.encoder.summary(**kwargs)
        self.decoder.summary(**kwargs)
