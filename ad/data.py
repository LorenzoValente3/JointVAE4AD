
import numpy as np
import tensorflow as tf

from ad.constants import MAX_CLIPPED_ENERGY
from ad import utils


def normalize(x: np.ndarray) -> np.ndarray:
    return np.clip(x / MAX_CLIPPED_ENERGY, a_min=0.0, a_max=1.0)


def load_npz(path):
    npz = np.load(path.numpy())

    image = np.array(npz['arr_0'], dtype=np.float32)
    return normalize(image)


def load_wrapper(path):
    images = tf.py_function(load_npz, inp=[path], Tout=[tf.float32])
    return images[0]


def make_dataset(folder: str, batch_size: int, buffer=1024, prefetch=1,
                 seed=utils.SEED, parallel_calls=2, shuffle=True) -> tf.data.Dataset:
    ds = tf.data.Dataset.list_files(folder + '/*.npz', shuffle=shuffle,
                                    seed=seed)

    ds = ds.map(load_wrapper, num_parallel_calls=parallel_calls)
    ds = ds.unbatch()

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    if (prefetch >= 1) or (prefetch == tf.data.AUTOTUNE):
        return ds.prefetch(int(prefetch))

    return ds


def downsampled_dataset(dataset: tf.data.Dataset, kernel_size: int, calls=tf.data.AUTOTUNE,
                        padding='same') -> tf.data.Dataset:
    """Use dept-wise convolutions to down-sample while preserving the total energy"""
    assert kernel_size >= 2

    filter_ = tf.ones((kernel_size, kernel_size, 3, 1))
    strides = [1, kernel_size, kernel_size, 1]

    return dataset.map(lambda x: tf.nn.depthwise_conv2d(x, filter_, strides, padding=padding.upper()),
                       num_parallel_calls=int(calls))


def supervised_dataset(folder: str, batch_size: int, buffer=2048,
                       seed=utils.SEED, parallel_calls=4, shuffle=True) -> tf.data.Dataset:
    def load(path):
        path = path.numpy()
        npz = np.load(path)

        image = np.array(npz['arr_0'], dtype=np.float32)

        if 'qcd' in str(path):
            labels = np.zeros((len(image), 1), dtype=np.float32)
        else:
            labels = np.ones((len(image), 1), dtype=np.float32)

        return normalize(image), labels

    def wrapper(path):
        images, labels = tf.py_function(load, inp=[path],
                                        Tout=[tf.float32, tf.float32])
        return images, labels

    qcd = tf.data.Dataset.list_files(folder + '/qcd/*.npz')
    suep = tf.data.Dataset.list_files(folder + '/suep/*.npz')

    ds = qcd.concatenate(suep)

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(wrapper, num_parallel_calls=parallel_calls)
    ds = ds.unbatch()

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_supervised(folder: str, batch_size: int, buffer=1024, prefetch=1, should_normalize=True,
                    seed=utils.SEED, parallel_calls=2, down_sample: dict = None, shuffle=True) -> tf.data.Dataset:
    def load_numpy(path):
        npz = np.load(path.numpy())

        image = np.array(npz['images'], dtype=np.float32)
        label = np.array(npz['labels'] != 0, dtype=np.float32)

        if should_normalize:
            return normalize(image), label

        return image, label

    def wrapper(path):
        images, labels = tf.py_function(load_numpy, inp=[path], Tout=[tf.float32, tf.float32])
        return images, labels

    ds = tf.data.Dataset.list_files(folder + '/*.npz', shuffle=shuffle, seed=seed)

    ds = ds.map(wrapper, num_parallel_calls=parallel_calls)
    ds = ds.unbatch()

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    if isinstance(down_sample, dict):
        kernel_size = down_sample.get('kernel_size', 5)
        padding = down_sample.get('padding', 'same').upper()

        filter_ = tf.ones((kernel_size, kernel_size, 3, 1))
        strides = [1, kernel_size, kernel_size, 1]

        def conv_fn(x):
            x = tf.nn.depthwise_conv2d(x, filter_, strides, padding=padding)

            if down_sample.get('normalize', False):
                x = tf.reduce_sum(x, axis=-1, keepdims=True)
                x /= tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True)

            return x

        ds = ds.map(lambda x, y: (conv_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if (prefetch >= 1) or (prefetch == tf.data.AUTOTUNE):
        return ds.prefetch(int(prefetch))

    return ds


# -----------------------------------------------------------------------------------------------------
# -- New utils on down-sampled dataset
# -----------------------------------------------------------------------------------------------------

CLIP_MIN = 0.0799 / 2000.0
CLIP_MAX = 2000.0


def clip(x):
    return np.clip(x, 0.0, CLIP_MAX) / CLIP_MAX
