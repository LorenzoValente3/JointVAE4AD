
import os
import ad.constants

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from tensorflow.keras.layers import *

from typing import List

from ad import utils
from ad import metrics
from ad import layers


class DualAE(tf.keras.Model):
    # constants
    MASSES = [125, 400, 700, 1000]
    CLIP_MIN = 0.0799 / 2000.0
    CLIP_MAX = 2000.0

    def __init__(self, latent_size: int, alpha=1.0, beta=1.0, gamma=0.0, name=None,
                 class_weights=(0.1, 10.0), threshold=None, **kwargs):
        kwargs = kwargs or {}
        self._base_model_initialized = True  # "hack" to avoid annoying error about subclassing tf.keras.Model
        # assert latent_size >= 1
        
        if threshold is None:
            threshold = self.CLIP_MIN
        
        self.latent_size = latent_size
        # self.latent_size = int(latent_size)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.threshold = tf.constant(threshold, dtype=tf.float32)

        # Coefficients to weight the loss function:
        self.alpha = tf.constant(float(alpha), dtype=tf.float32)
        self.beta = tf.constant(float(beta), dtype=tf.float32)
        self.gamma = tf.constant(float(gamma), dtype=tf.float32)

        self.bce = tf.losses.BinaryCrossentropy(reduction='none', axis=[])

        # build the encoder and decoder networks
        enc_args = kwargs.pop('encoder', {})

        self.encoder1 = self.get_encoder(self.latent_size, **enc_args)
        self.encoder2 = self.get_encoder(self.latent_size, **enc_args)
        self.decoder = self.get_decoder(**kwargs.pop('decoder', {}))

        self.concat = Concatenate(name='concat-z')

        if isinstance(self.latent_size, (int, float)):
            self.project = Dense(units=self.latent_size, name='project')
        else:
            self.project = Conv2D(filters=self.latent_size[0], kernel_size=1, name='conv-project')

        # define metrics to track losses, grads, and weights
        self.trackers = dict(loss=tf.keras.metrics.Mean(name='total_loss'),
                             ssim=ad.metrics.SSIM(name='ssim'),
                             mse=ad.metrics.MSE(name='mse'),
                             mse_energy=ad.metrics.MSE(name='mse_e'),
                             mse_mask=ad.metrics.MSE(name='mse_m'),
                             psnr=ad.metrics.PSNR(name='psnr'),
                             psnr_energy=ad.metrics.PSNR(name='psnr_e'),
                             psnr_mask=ad.metrics.PSNR(name='psnr_m'),
                             cosine_sim=tf.keras.metrics.Mean(name='cosine_similarity'),
                             reconstr_loss_energy=tf.keras.metrics.Mean(name='reconstruction_loss_e'),
                             reconstr_loss_mask=tf.keras.metrics.Mean(name='reconstruction_loss_m'),
                             true_energy=tf.keras.metrics.Mean(name='true_energy'),
                             pred_energy=tf.keras.metrics.Mean(name='pred_energy'),
                             grads_norm=tf.keras.metrics.Mean(name='gradients_norm'),
                             weights_norm=tf.keras.metrics.Mean(name='weights_norm'))

        self.test_trackers = [k for k in self.trackers.keys() if '_norm' not in k]

        super().__init__(name=name)

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return list(self.trackers.values())

    def call(self, x: tf.Tensor, **kwargs):
        z1 = self.encoder1(x, **kwargs)
        z2 = self.encoder2(tf.cast(x > 0.0, dtype=x.dtype), **kwargs)

        z = self.project(self.concat([z1, z2]))
        ye, ym = self.decoder(z, **kwargs)

        return self.threshold_fn(ye * ym)

    def get_encoder(self, latent_size: int, conv_filters: List[int], groups=None,
                    input_shape=(72, 58, 1), activation='relu', kernel=3,
                    include_se=False, dropout=0, **kwargs) -> tf.keras.Model:
        """Defines the architecture of the Encoder"""
        def activation_fn(layer: Layer, block: str):
            h = BatchNormalization(name=f'batch_norm-b{block}')(layer)
            return Activation(activation, name=f'activation-b{block}')(h)

        def conv_block(layer: Layer, filters: int, block: int):
            h = Conv2D(filters, kernel_size=kernel, padding='same', groups=groups,
                       name=f'conv-b{block}', activation=activation, **kwargs)(layer)

            h = SeparableConv2D(filters // 2, kernel_size=kernel, padding='same',
                                name=f'sep_conv-b{block}', **kwargs)(h)
            h = activation_fn(h, f'{block}_0')

            if include_se:
                h = ad.layers.SqueezeAndExcite(activation=activation, name=f'SE-b{block}',
                                               **kwargs)(h)
            # overlapping Max-pooling
            return MaxPool2D(pool_size=2, strides=2, padding='same',
                             name=f'max_pool-b{block}')(h)

        inputs = Input(shape=input_shape, name='images')

        x = Conv2D(filters=conv_filters[0], kernel_size=kernel, strides=2,
                   padding='same', activation=activation, name='conv-stem',
                   **kwargs)(inputs)

        for i, num_filters in enumerate(conv_filters[1:]):
            if dropout > 0.0:
                x = SpatialDropout2D(rate=float(dropout), name=f'dropout-b{i}')(x)

            x = conv_block(x, filters=int(num_filters), block=i + 1)

        x = Flatten(name='flatten')(x)
        z = Dense(units=latent_size, name='z')(x)

        return tf.keras.Model(inputs, outputs=z, name='Encoder')

    def get_decoder(self, conv_filters: List[int], shape=(5, 4), kernel=3, groups=2,
                    activation='relu', crop=(72, 58),
                    out_filters=1, bias=0.0, **kwargs) -> tf.keras.Model:
        """Defines the architecture of the Decoder"""
        def up_sample(layer: Layer, filters: int, block: int):
            h = UpSampling2D(name=f'upsample-b{block}')(layer)
            h = Conv2D(filters, kernel_size=kernel, name=f'conv-b{block}',
                       groups=groups, padding='same', activation=activation,
                       **kwargs)(h)
            h = SeparableConv2D(filters // 2, kernel_size=kernel,
                                name=f'sep_conv-b{block}', padding='same', **kwargs)(h)

            h = BatchNormalization(name=f'batchnorm-b{block}')(h)
            return Activation(activation, name=f'activation-b{block}')(h)

        latent = Input(shape=(self.latent_size,), name='latent')

        x = ad.layers.SpatialBroadcast(width=shape[1], height=shape[0])(latent)
        x.set_shape((None, shape[0], shape[1], latent.shape[-1] + 2))

        x = Conv2D(filters=conv_filters[0], kernel_size=kernel, padding='same',
                   name='conv-z', **kwargs)(x)
        x = BatchNormalization(name='batchnorm-z')(x)
        x = Activation(activation, name='activation-z')(x)

        for i, num_filters in enumerate(conv_filters[1:]):
            x = up_sample(x, filters=int(num_filters), block=i + 1)

        x = CenterCrop(*crop, name='crop')(x)

        energy = Conv2D(filters=out_filters, kernel_size=kernel, padding='same',
                        bias_initializer=tf.keras.initializers.Constant(bias),
                        name='energy', activation=tf.nn.sigmoid)(x)

        mask = Conv2D(filters=out_filters, kernel_size=kernel, padding='same',
                      name='mask', activation=tf.nn.sigmoid)(x)

        return tf.keras.Model(inputs=latent, outputs=[energy, mask], name='Decoder')

    @tf.function
    def threshold_fn(self, x: tf.Tensor):
        mask = tf.cast(tf.greater(x, self.threshold), dtype=tf.float32)
        return x * mask

    @tf.function
    def train_step(self, batch: tf.Tensor):
        mask = batch > 0.0
        weights = tf.where(mask, x=self.class_weights[1], y=self.class_weights[0])
        mask = tf.cast(mask, dtype=batch.dtype)

        with tf.GradientTape() as tape:
            z_e = self.encoder1(batch, training=True)
            z_m = self.encoder2(mask, training=True)

            z = self.project(self.concat([z_e, z_m]))

            pred_energy, pred_mask = self.decoder(z, training=True)
            pred_energy = self.threshold_fn(pred_energy)

            # compute the total loss
            loss_energy = self.alpha * self.bce_loss(pred_energy, batch, weights)
            loss_energy += self.beta * self.dice_loss(pred_energy, batch)

            loss_mask = self.bce_loss(pred_mask, mask)
            loss_mask += self.beta * self.dice_loss(pred_mask, mask)

            latent_loss = self.cosine_similarity(z_e, z_m)
            reconstruction_loss = loss_energy + loss_mask

            reg_loss = tf.reduce_sum(self.losses)
            total_loss = reconstruction_loss + reg_loss + self.gamma * latent_loss

        # compute the gradients of the `total_loss` w.r.t. the networks parameters
        grads = tape.gradient(total_loss, self.trainable_variables)
        trainable_vars = self.trainable_variables

        # take a gradient step that updates the weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # update metrics and return their state
        reco = pred_energy * pred_mask

        self.update_trackers(loss=total_loss, reconstr_loss_mask=loss_mask,
                             reconstr_loss_energy=loss_energy, cosine_sim=latent_loss,
                             mse=(reco, batch), psnr=(reco, batch),
                             mse_energy=(pred_energy, batch),
                             mse_mask=(pred_mask, mask), ssim=(reco, batch),
                             psnr_energy=(pred_energy, batch),
                             psnr_mask=(pred_mask, mask),
                             true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(reco, axis=[1, 2, 3]),
                             grads_norm=utils.tf_global_norm(grads),
                             weights_norm=utils.tf_global_norm(trainable_vars))

        return {k: metric.result() for k, metric in self.trackers.items()}

    @tf.function
    def test_step(self, batch: tf.Tensor):
        mask = batch > 0.0
        weights = tf.where(mask, x=self.class_weights[1], y=self.class_weights[0])
        mask = tf.cast(mask, dtype=batch.dtype)

        z_e = self.encoder1(batch, training=False)
        z_m = self.encoder2(mask, training=False)

        z = self.project(self.concat([z_e, z_m]))

        pred_energy, pred_mask = self.decoder(z, training=False)
        pred_energy = self.threshold_fn(pred_energy)
        reco = pred_energy * pred_mask

        # compute losses
        loss_energy = self.alpha * self.bce_loss(pred_energy, batch, weights)
        loss_energy += self.beta * self.dice_loss(pred_energy, batch)

        loss_mask = self.bce_loss(pred_mask, mask)
        loss_mask += self.beta * self.dice_loss(pred_mask, mask)

        latent_loss = self.cosine_similarity(z_e, z_m)
        reconstruction_loss = loss_energy + loss_mask

        reg_loss = tf.reduce_sum(self.losses)
        total_loss = reconstruction_loss + reg_loss + self.gamma * latent_loss

        self.update_trackers(loss=total_loss, reconstr_loss_mask=loss_mask,
                             reconstr_loss_energy=loss_energy, cosine_sim=latent_loss,
                             mse=(reco, batch), psnr=(reco, batch),
                             mse_energy=(pred_energy, batch),
                             mse_mask=(pred_mask, mask), ssim=(reco, batch),
                             psnr_energy=(pred_energy, batch),
                             psnr_mask=(pred_mask, mask),
                             true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(reco, axis=[1, 2, 3]))

        return {k: self.trackers[k].result() for k in self.test_trackers}

    def update_trackers(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, tuple):
                self.trackers[k].update_state(*v)

            elif isinstance(v, dict):
                self.trackers[k].update_state(**v)
            else:
                self.trackers[k].update_state(v)

    @tf.function
    def bce_loss(self, y_pred, y_true, weights=None):
        loss = self.bce(y_true, y_pred, sample_weight=weights)
        loss = tf.reduce_sum(loss, axis=[1, 2, 3])
        return tf.reduce_mean(loss)

    @tf.function
    def dice_loss(self, y_pred, y_true):
        # Source: https://arxiv.org/pdf/1807.10097v1.pdf (page 6)
        sum_p = tf.reduce_sum(tf.square(y_pred), axis=[1, 2, 3])
        sum_t = tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3])

        union = sum_p + sum_t
        intersection = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])

        loss = union / (2.0 * intersection)
        return tf.reduce_sum(loss)

    @tf.function
    def cosine_similarity(self, p, z):
        p = tf.nn.l2_normalize(p, axis=1)
        z = tf.nn.l2_normalize(z, axis=1)

        return tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

    @classmethod
    def clip(cls, x):
        """Data pre-processing"""
        return np.clip(x, 0.0, cls.CLIP_MAX) / cls.CLIP_MAX

    def summary(self):
        self.encoder1.summary()
        self.decoder.summary()


class ConvAE(tf.keras.Model):
    def __init__(self, name=None, max_grad_norm: float = None, **kwargs):
        self._base_model_initialized = True
        self.latent_size = None

        if isinstance(max_grad_norm, (int, float)):
            self.should_clip_grads = True
            self.grad_norm = tf.constant(max_grad_norm, dtype=tf.float32)
        else:
            self.should_clip_grads = False

        # build the encoder and decoder networks
        self.encoder = self.get_encoder(**kwargs.pop('encoder', {}))
        self.decoder = self.get_decoder(latent_shape=self.encoder.output.shape[1:],
                                        **kwargs.pop('decoder', {}))

        # define metrics to track losses, grads, and weights
        self.trackers = dict(loss=tf.keras.metrics.Mean(name='total_loss'),
                             mse=ad.metrics.MSE(name='mse'),
                             psnr=ad.metrics.PSNR(name='psnr'),
                             ssim=ad.metrics.SSIM(name='ssim'),
                             true_energy=tf.keras.metrics.Mean(name='true_energy'),
                             pred_energy=tf.keras.metrics.Mean(name='pred_energy'),
                             grads_norm=tf.keras.metrics.Mean(name='gradients_norm'),
                             weights_norm=tf.keras.metrics.Mean(name='weights_norm'))

        self.test_trackers = [k for k in self.trackers.keys() if '_norm' not in k]
        super().__init__(name=name)

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return list(self.trackers.values())

    def call(self, x, **kwargs):
        z = self.encoder(x, **kwargs)
        return self.decoder(z, **kwargs)

    def get_encoder(self, input_shape: tuple, depths: List[int], filters: List[int],
                    activation=tf.nn.relu6, kernel=3, groups=None, other_layers=None,
                    **kwargs) -> tf.keras.Model:
        assert len(depths) == len(filters)

        images = Input(shape=input_shape, name='image')
        x = images

        for j, depth in enumerate(depths):
            x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, stride=2, **kwargs,
                                    activation=activation, name=f'dconv-b{j}')(x)

            # add residual blocks
            for i in range(depth):
                r = x  # residual

                x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs, groups=groups,
                                        activation=activation, name=f'conv1-b{j}_{i}')(x)
                x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs, groups=groups,
                                        activation=activation, name=f'conv2-b{j}_{i}')(x)

                x = Add(name=f'add-b{j}_{i}')([x, r])

        if isinstance(other_layers, (list, tuple)):
            for layer in other_layers:
                x = layer(x)

        z = x
        return tf.keras.Model(inputs=images, outputs=z, name='Res-Encoder')

    def get_decoder(self, latent_shape: tuple, depths: List[int], filters: List[int],
                    crop: tuple, activation=tf.nn.relu6, kernel=3, size=(5, 4, 256),
                    out_channels=1, groups=None, **kwargs) -> tf.keras.Model:
        assert len(depths) == len(filters)

        latents = Input(shape=latent_shape, name='z')

        if len(latent_shape) == 1:
            x = ad.layers.SpatialBroadcast(width=size[1], height=size[0], name='spatial-broadcast')(latents)
            x.set_shape((None, size[0], size[1], latent_shape[-1] + 2))

            x = ad.layers.ConvLayer(filters=size[-1], kernel=kernel, name='conv-expand',
                                    activation=activation, **kwargs)(x)
        else:
            x = latents

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
    def train_step(self, batch: tf.Tensor):
        with tf.GradientTape() as tape:
            z = self.encoder(batch, training=True)
            x = self.decoder(z, training=True)

            loss = self.compiled_loss(x, batch)

        weights = self.trainable_variables
        grads = tape.gradient(loss, weights)

        if self.should_clip_grads:
            grads = tf.clip_by_global_norm(grads, clip_norm=self.grad_norm)

        self.optimizer.apply_gradients(zip(grads, weights))

        self.update_trackers(loss=loss, mse=(x, batch), psnr=(x, batch), ssim=(x, batch),
                             true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]),
                             grads_norm=utils.tf_global_norm(grads),
                             weights_norm=utils.tf_global_norm(weights))

        return {k: metric.result() for k, metric in self.trackers.items()}

    @tf.function
    def test_step(self, batch: tf.Tensor):
        z = self.encoder(batch, training=False)
        x = self.decoder(z, training=False)

        loss = self.compiled_loss(x, batch)

        self.update_trackers(loss=loss, mse=(x, batch), psnr=(x, batch), ssim=(x, batch),
                             true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]))

        return {k: self.trackers[k].result() for k in self.test_trackers}

    @staticmethod
    def bce_loss(y_pred, y_true):
        bce = tf.keras.losses.BinaryCrossentropy(axis=[], reduction='none')

        loss = bce(y_true, y_pred)
        loss = tf.reduce_sum(loss, axis=[1, 2, 3])
        return tf.reduce_mean(loss)

    @staticmethod
    def ssim_loss(y_pred, y_true):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        return -ssim

    def update_trackers(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, tuple):
                self.trackers[k].update_state(*v)

            elif isinstance(v, dict):
                self.trackers[k].update_state(**v)
            else:
                self.trackers[k].update_state(v)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
