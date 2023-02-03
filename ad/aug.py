"""Data augmentations"""

import tensorflow as tf

from ad import utils


def flip_eta(x):
    return tf.cast(tf.image.flip_left_right(x), dtype=tf.float32)


def tf_rotate_phi_down(x, amount: int):
    h = tf.shape(x)[0]

    # slice the input
    bottom = x[h - amount:, :, :]
    top = x[:h - amount, :, :]

    return tf.cast(tf.concat([bottom, top], axis=0), dtype=tf.float32)


def tf_rotate_phi_up(x, amount: int):
    # slice the input
    top = x[:amount, :, :]
    bottom = x[amount:, :, :]

    return tf.cast(tf.concat([bottom, top], axis=0), dtype=tf.float32)


def tf_augment(x, delta=8, size=7):
    # we have 6 possible choices
    choice = utils.tf_random_choice(size=6)
    amount = delta * (1 + utils.tf_random_choice(size=int(size)))  # up-to amount of 56
    amount = tf.cast(amount, dtype=tf.int32)

    if choice == 1:
        # flip in eta
        return flip_eta(x)

    if choice == 2:
        # downward rotation in phi
        return tf_rotate_phi_down(x, amount=amount)

    if choice == 3:
        # upward rotation in phi
        return tf_rotate_phi_up(x, amount=amount)

    if choice == 4:
        # flip + down rotation
        return tf_rotate_phi_down(flip_eta(x), amount=amount)

    if choice == 5:
        # flip + up rotation
        return tf_rotate_phi_up(flip_eta(x), amount=amount)

    # no augmentation
    return tf.cast(x, dtype=tf.float32)


def augmented_dataset(tensors, batch_size: int, buffer=2**13):
    """Data augmentations for the unsupervised dataset (x,)"""
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    ds = ds.shuffle(buffer_size=int(buffer), seed=utils.SEED, reshuffle_each_iteration=True)
    ds = ds.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=int(batch_size), num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.repeat(count=6)
    return ds.prefetch(tf.data.AUTOTUNE)


def augmented_dataset2(*tensors, batch_size: int, buffer=2**13):
    """Data augmentations for the supervised dataset (x, y)"""
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    ds = ds.shuffle(buffer_size=int(buffer), seed=utils.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=int(batch_size), num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.repeat(count=6)
    return ds.prefetch(tf.data.AUTOTUNE)
