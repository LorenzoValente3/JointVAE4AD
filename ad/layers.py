import torch
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers as tfkl
from typing import List


class Depth2Space(tf.keras.layers.Lambda):
    def __init__(self, ratio=2, name=None, **kwargs):
        assert ratio >= 2
        super().__init__(lambda t: tf.nn.depth_to_space(t, block_size=int(ratio)),
                         name=name, **kwargs)


class Space2Depth(tf.keras.layers.Lambda):
    def __init__(self, ratio=2, name=None, **kwargs):
        assert ratio >= 2
        super().__init__(lambda t: tf.nn.space_to_depth(t, block_size=int(ratio)),
                         name=name, **kwargs)


class Sampling(tf.keras.layers.Layer):
    """Sampling layer for VAE that implements the re-parametrization trick for differentiable Gaussian sampling"""

    @tf.function
    def call(self, inputs, **kwargs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]

        # we apply the multigaussian noise to every points at once
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # Re-parametrization trick
        return mean + tf.exp(0.5 * log_var) * epsilon

class joint_sampling(tf.keras.layers.Layer):
    """Sampling layer for joint_VAE that implements the re-parametrization trick for differentiable Gaussian sampling"""
    def __init__(self, temp = 1, eps = 1e-7, name=None, **kwargs):
        super().__init__(name=name)

        self.temperature = temp
        self.eps = eps

    @tf.function
    def call(self, inputs, **kwargs) :
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        inputs -> :param mu:      (Tensor) mean of the latent Gaussian  [B x D]
        inputs -> :param log_var: (Tensor) Log variance of the latent Gaussian [B x D]
        inputs -> :param q:       (Tensor) Categorical latent Codes [B x Q]
        :return:  (Tensor) [B x (D + Q)]
        """
        mu, log_var, q = inputs

        std = tf.exp(0.5 * log_var)
        e = tf.keras.backend.random_normal(shape = tf.shape(std))
        z = e * std + mu

        # Sample from Gumbel
        u = tf.random.uniform(shape = tf.shape(q)) 
        g = - tf.math.log(- tf.math.log(u + self.eps) + self.eps)

        # Gumbel-Softmax sample
        s = tf.nn.softmax(( q + g ) / self.temperature, axis=-1)
        s = tf.reshape(s, shape =tf.shape(q))

        return tf.concat([z, s], axis=1, name='concat')

class SqueezeAndExcite(tf.keras.layers.Layer):
    """Based on https://github.com/titu1994/keras-squeeze-excite-network"""

    def __init__(self, ratio=16, activation='relu', name=None, **kwargs):
        super().__init__(name=name)

        self.ratio = int(ratio)
        self.kwargs = kwargs
        self.activation = activation

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = None
        self.dense1 = None
        self.dense2 = None
        self.multiply = tf.keras.layers.Multiply()

    def build(self, input_shape: tuple):
        num_filters = input_shape[-1]

        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, num_filters))
        self.dense1 = tf.keras.layers.Dense(units=num_filters // self.ratio, use_bias=False,
                                            activation=self.activation, **self.kwargs)
        self.dense2 = tf.keras.layers.Dense(units=num_filters, activation='sigmoid',
                                            use_bias=False, **self.kwargs)

    @tf.function
    def call(self, x: tf.Tensor):
        h = self.global_pool(x)
        h = self.reshape(h)

        h = self.dense1(h)
        h = self.dense2(h)

        return self.multiply([x, h])


class SpatialBroadcast(tf.keras.layers.Layer):
    """A layer that implements the 'spatial broadcast' operation used in VAE decoder networks.
        - Spatial Broadcast Decoder: https://arxiv.org/pdf/1901.07017
    """

    def __init__(self, width: int, height: int, **kwargs):
        w = int(width)
        h = int(height)

        assert w > 1 and h > 1
        super().__init__(**kwargs)

        self.w = w
        self.h = h

        # create coordinates that will later be concatenated to the tiled latents
        self.tile_shape = (1, h, w, 1)
        self.x_mesh, self.y_mesh = self.get_xy_meshgrid(w, h)

    def call(self, latents, **kwargs):
        batch_size = tf.shape(latents)[0]

        # tile the latent vectors
        z = tf.reshape(latents, shape=(batch_size, 1, 1, -1))
        z = tf.tile(z, multiples=self.tile_shape)

        # also tile the xy-meshgrid
        x = tf.tile(self.x_mesh, multiples=(batch_size, 1, 1, 1))
        y = tf.tile(self.y_mesh, multiples=(batch_size, 1, 1, 1))

        # lastly concatenate along the channel axis
        return tf.concat([z, x, y], axis=-1)

    def get_xy_meshgrid(self, w: int, h: int):
        x_coord = tf.linspace(-1, 1, w)
        y_coord = tf.linspace(-1, 1, h)

        # meshgrid & cast
        x, y = tf.meshgrid(x_coord, y_coord)
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        # expand shape (to 4D) to later match the tiled latents
        x = tf.reshape(x, shape=self.tile_shape)
        y = tf.reshape(y, shape=self.tile_shape)
        return x, y

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.update(width=self.w, height=self.h)
        return config


class ConvLayer(tfkl.Layer):
    """Convolutional layer with instance-normalization, activation, and optional dropout"""
    
    def __init__(self, filters: int, kernel: int, stride=1, name=None,
                 activation=tf.nn.relu6, dropout=0.0, **kwargs):
        super().__init__(name=name)

        self.conv = tfkl.Conv2D(filters=filters, kernel_size=kernel, strides=stride,
                                padding='same', **kwargs)
        self.norm = tfa.layers.InstanceNormalization()
        self.act = tfkl.Activation(activation)

        if dropout > 0.0:
            self.dropout = SpatialDropout2D(rate=float(dropout))
        else:
            self.dropout = None

    @property
    def layers(self) -> List[tfkl.Layer]:
        return [self.conv, self.norm, self.act]

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.norm(x, **kwargs)
        x = self.act(x)

        if self.dropout is not None:
            return self.dropout(x, **kwargs)

        return x


class UpConvLayer(tfkl.Layer):
    """Layer with bilinear-upsampling, convolution, instance-norm, and activation"""
    def __init__(self, filters: int, kernel: int, name=None,
                 activation=tf.nn.relu6, **kwargs):
        super().__init__(name=name)

        self.up_sample = tfkl.UpSampling2D(interpolation='bilinear')
        self.conv = tfkl.Conv2D(filters=filters, kernel_size=kernel,
                                padding='same', **kwargs)
        self.norm = tfa.layers.InstanceNormalization()
        self.act = tfkl.Activation(activation)

    @property
    def layers(self) -> List[tfkl.Layer]:
        return [self.up_sample, self.conv, self.norm, self.act]

    def call(self, x, **kwargs):
        x = self.up_sample(x)
        x = self.conv(x)
        x = self.norm(x, **kwargs)
        return self.act(x)