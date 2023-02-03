import os
import ad.constants

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import *
from qkeras import *
from qkeras.quantizers import *
from tensorflow.keras.regularizers import l1, l2

from typing import List

from ad import utils
from ad import metrics
from ad import layers



class QJointVAE(keras.Model):

    """ Quantized Joint VAE model"""      
  
    def __init__(self, 
                 latent_dim: int = 32, 
                 temperature: float = 50.,
                 categorical_dim: int = 16, 
                 alpha: float = 1.,
                 beta: float = 3e3, 
                 eps_kl: float = 1e-7,  
                 name = None, 
                 **kwargs):
        super().__init__(name = name)


        #parameter assignment
        self.latent_dim = latent_dim
        self.temp = temperature #gumbel-softmax
        self.categorical_dim = categorical_dim #gumbel-softmax
        self.alpha = alpha
        self.beta = beta
        self.eps_kl = eps_kl # KL divergence between gumbel-softmax distribution


        # build the encoder and decoder networks
        self.encoder = self.build_encoder(**kwargs.pop('encoder', {}))
        self.decoder = self.build_decoder(latent_shape= self.encoder.output[-1].shape[1:] ,
                                        **kwargs.pop('decoder', {}))
        
        #implementing the loss fct
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_cont_loss_tracker = keras.metrics.Mean(name="kl_cont_loss")
        self.kl_disc_loss_tracker = keras.metrics.Mean(name="kl_disc_loss")

        self.mse_tracker = ad.metrics.MSE(name="mse")
        self.psnr_tracker = ad.metrics.PSNR(name='psnr')
        self.ssim_tracker = ad.metrics.SSIM(name='ssim')

    @property
    def metrics(self)  -> List[tf.keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_cont_loss_tracker,
            self.kl_disc_loss_tracker ,
            self.mse_tracker,
            self.psnr_tracker,
            self.ssim_tracker,
        ]
    
    def call(self, x, **kwargs):
        q, mean, var, z= self.encoder(x, **kwargs)
        return self.decoder(z, **kwargs)
    
    def build_encoder(self, input_shape: tuple, depths: List[int], filters: List[int],
                    kernel=3, groups=None, 
                    qconv= "quantized_bits(16,6,alpha=1)",
                    activation = tf.nn.leaky_relu ,
                    # activation=quantized_relu(16,negative_slope=0.25), # negative_slope for leaky relu
                    **kwargs) -> tf.keras.Model:
        """Building the encoder architecture for the variational autoencoder. 
        The final encoding dimension can be chosen."""

        assert len(depths) == len(filters)

        images = Input(shape = input_shape, name='image')
        x = images

        for j, depth in enumerate(depths):
            x = QConv2D(filters=filters[j], kernel_size=kernel, strides=2, groups=groups if j > 0 else 1,
                                padding='same', 
                                # kernel_regularizer=l1(0.0001), 
                                **kwargs, name=f'qdconv-b{j}')(x)
            x = tfa.layers.InstanceNormalization(name=f'bn-b{j}')(x) #prova instance normalization
            x = Activation(activation, name=f'activ-b{j}')(x)

            # add residual blocks
            for i in range(depth):
                r = x  # residual

                x = QConv2D(filters=filters[j], kernel_size=kernel, strides=1,groups=groups,
                                padding='same', 
                                # kernel_regularizer=l1(0.0001), 
                                **kwargs, name=f'qconv1-b{j}_{i}')(x)
                x = tfa.layers.InstanceNormalization(name=f'bn1-b{j}_{i}')(x)
                x = Activation(activation, name=f'activ1-b{j}_{i}')(x)

                x = QConv2D(filters=filters[j], kernel_size=kernel, strides=1,groups=groups,
                                padding='same',
                                # kernel_regularizer=l1(0.0001), 
                                **kwargs, name=f'qconv2-b{j}_{i}')(x)
                x = tfa.layers.InstanceNormalization(name=f'bn2-b{j}_{i}')(x)
                x = Activation(activation, name=f'activ2-b{j}_{i}')(x)

                x = Add(name=f'add-b{j}_{i}')([x, r])


        z = Flatten()(x)

        q = Dense ( units = self.categorical_dim, name='z_categorical')(z)
        encoded_mean = Dense(units = self.latent_dim, name='z_mean')(z)
        encoded_var = Dense(units = self.latent_dim, name='z_var')(z)
        
        encoder_out = layers.joint_sampling(temp = self.temp, name='encoder_output')([encoded_mean, encoded_var, q])

        return tf.keras.Model(inputs = images, outputs=[q, encoded_mean, encoded_var, encoder_out], name='Res-Encoder')

    def build_decoder(self, latent_shape: tuple, depths: List[int], filters: List[int],
                    crop: tuple, activation=tf.nn.relu6, kernel=3, size=(5, 4, 256),
                    out_channels=1, groups=None, **kwargs) -> tf.keras.Model:
      """
      ## Build the decoder
      """
      assert len(depths) == len(filters)

      latents = Input(shape=latent_shape, name='z')

      if len(latent_shape) == 1:
          x = ad.layers.SpatialBroadcast(width=size[1], height=size[0], name='spatial-broadcast')(latents)
          x.set_shape((None, size[0], size[1], self.latent_dim + self. categorical_dim + 2))
          
          x = ad.layers.ConvLayer(filters=size[-1], kernel=kernel, name='conv-expand',
                                    activation=activation, **kwargs)(x)
      else:
          x = Reshape((5 * 4 * 256))(latents)

      for j, depth in enumerate(depths):
          x = ad.layers.UpConvLayer(filters=filters[j], kernel=kernel, **kwargs,
                                      activation=activation, name=f'up_conv-b{j}')(x)

            # add residual blocks
          for i in range(depth):
              r = x  # residual

              x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs, groups=groups,
                                        activation=activation, name=f'conv1-b{j}_{i}')(x)
              x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs, groups=groups,
                                        activation=activation, name=f'conv2-b{j}_{i}')(x)

              x = Add(name=f'add-b{j}_{i}')([x, r])

        # reconstruction
      reco = CenterCrop(*crop, name='crop')(x)
      reco = Conv2D(filters=int(out_channels), kernel_size=kernel, padding='same',
                      activation=tf.nn.sigmoid, name='conv-reco')(reco)

      return tf.keras.Model(inputs=latents, outputs=reco, name='Res-Decoder')    

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            q, z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            
            #######################
            #######   RECO   ######
            #######################

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                ) * self.alpha
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
            h2 = q_p * tf.math.log(1. / self.categorical_dim + self.eps_kl)
            kl_disc_loss = tf.reduce_mean(tf.reduce_sum(h1- h2 , axis = 1 ) * self.beta, axis = 0)
            # kl_disc_loss = abs(kl_disc_loss)

            total_loss = reconstruction_loss + kl_cont_loss + kl_disc_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_cont_loss_tracker.update_state(kl_cont_loss)
        self.kl_disc_loss_tracker.update_state(kl_disc_loss)
        self.mse_tracker.update_state(reconstruction, data)
        self.psnr_tracker.update_state(reconstruction, data)
        self.ssim_tracker.update_state(reconstruction, data)

        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_cont_loss": self.kl_cont_loss_tracker.result(),
            "kl_disc_loss": self.kl_disc_loss_tracker.result(),
            "mse": self.mse_tracker.result(),
            "psnr": self.psnr_tracker.result(),
            "ssim": self.ssim_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        q, z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
            
        #######################
        #######   RECO   ######
        #######################

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
              ) * self.alpha
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
        h2 = q_p * tf.math.log(1. / self.categorical_dim + self.eps_kl)
        kl_disc_loss = tf.reduce_mean(tf.reduce_sum(h1- h2 , axis = 1 ) * self.beta, axis = 0)
        # kl_disc_loss = abs(kl_disc_loss)

        total_loss = reconstruction_loss + kl_cont_loss + kl_disc_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_cont_loss_tracker.update_state(kl_cont_loss)
        self.kl_disc_loss_tracker.update_state(kl_disc_loss)
        self.mse_tracker.update_state(reconstruction, data)
        self.psnr_tracker.update_state(reconstruction, data)
        self.ssim_tracker.update_state(reconstruction, data)

        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_cont_loss": self.kl_cont_loss_tracker.result(),
            "kl_disc_loss": self.kl_disc_loss_tracker.result(),
            "mse": self.mse_tracker.result(),
            "psnr": self.psnr_tracker.result(),
            "ssim": self.ssim_tracker.result(),
        }

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()  