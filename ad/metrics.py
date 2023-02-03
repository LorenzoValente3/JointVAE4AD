import numpy as np
import tensorflow as tf


class SSIM(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, image_range=1.0, **kwargs):
        super().__init__(lambda x, y: tf.image.ssim(x, y, max_val=float(image_range)), **kwargs)


class PSNR(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, image_range=1.0, **kwargs):
        super().__init__(lambda x, y: tf.image.psnr(x, y, max_val=float(image_range)), **kwargs)


class EnergySumMetric(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, stats: np.ndarray, **kwargs):
        self.stats = tf.constant(stats, dtype=tf.float32)

        super().__init__(fn=self.sum_energy, dtype=tf.float32, **kwargs)

    @tf.function
    def sum_energy(self, x: tf.Tensor, y: tf.Tensor):
        energy = tf.reduce_sum(x * self.stats, axis=[1, 2, 3])
        return tf.reduce_mean(energy)

    def update_state(self, x, *args, **kwargs):
        return super().update_state(x, x, *args, **kwargs)


class MSE(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, **kwargs):
        super().__init__(fn=self.compute, dtype=tf.float32, **kwargs)

    @staticmethod
    @tf.function
    def compute(x: tf.Tensor, y: tf.Tensor):
        """MSE computed for images: sum over spatial dimensions and average over batch"""
        sq_err = tf.square(x - y)
        error = tf.reduce_sum(sq_err, axis=[1, 2])
        return tf.reduce_mean(error)
