import os
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from typing import List, Union


SEED = None


def get_name(label: int) -> str:
    """Returns the name of the given class label"""
    from ad.constants import LABELS
    return LABELS[int(label)]


def get_bkg_name() -> str:
    from ad.constants import LABELS, BKG_INDEX
    return LABELS[BKG_INDEX]


def get_label_idx(label: str) -> int:
    from ad.constants import LABELS
    assert label in LABELS

    for i, name in enumerate(LABELS):
        if label == name:
            return i


def get_name_from(mass: int) -> str:
    from ad.constants import MASSES, LABELS, BKG_INDEX
    name = LABELS[BKG_INDEX]

    for k, masses in MASSES.items():
        if mass in masses.values():
            name = get_name(label=k)
            break

    return name


def get_masses(label: Union[int, str]) -> dict:
    if isinstance(label, str):
        label = get_label_idx(label)
    else:
        label = int(label)

    # if label == 0:
    #     return []

    from ad.constants import MASSES
    return MASSES[label]


def get_mass(label: Union[int, str], mass_idx: int) -> int:
    if isinstance(label, str):
        label = get_label_idx(label)

    if int(label) == 0:
        return 0

    return get_masses(label)[int(mass_idx)]


def set_random_seed(seed: int):
    """Sets the random seed for TensorFlow, numpy, python's random"""
    global SEED

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        SEED = seed
        print(f'Random seed {SEED} set.')


def tf_random_choice(size: int):
    return tf.argmax(tf.random.uniform(shape=(size,), maxval=1.0))


def get_random_generator(seed=SEED) -> np.random.Generator:
    """Returns a numpy random generator instance"""
    if seed is not None:
        seed = int(seed)
        assert 0 <= seed < 2 ** 32

    return np.random.default_rng(np.random.MT19937(seed=seed))


def tf_global_norm(values: List[tf.Tensor]):
    """Computes the global l2-norm of a list of tensors"""
    # Euclidean norm of each item in the provided list
    local_norms = [tf.norm(v) for v in values]

    # now compute the global l2-norm
    return tf.sqrt(tf.reduce_sum([norm * norm for norm in local_norms]))


def free_mem():
    return gc.collect()


def get_plot_axes(rows: int, cols: int, size=(12, 10), **kwargs):
    rows = int(rows)
    cols = int(cols)

    assert rows >= 1
    assert cols >= 1

    fig, axes = plt.subplots(nrows=rows, ncols=cols, **kwargs)

    fig.set_figwidth(size[0] * cols)
    fig.set_figheight(size[1] * rows)

    return axes


def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def read_npz(folder: str, dtype=np.float16, limit: int = None, shuffle=False,
             verbose=True, keys='arr_0', skip: str = None):
    """Reads all the .npz files in the given `folder`, returning one big array"""
    should_skip = isinstance(skip, str)
    files = os.listdir(folder)

    if shuffle:
        random.shuffle(files)

    limit = np.inf if limit is None else int(limit)
    count = 0

    if isinstance(keys, (list, tuple)):
        assert len(keys) > 0

        if len(keys) == 1:
            dataset = []
        else:
            dataset = {k: [] for k in keys}
    else:
        assert isinstance(keys, str)
        keys = [keys]
        dataset = []

    for i, file_name in enumerate(files):
        if should_skip and (skip in file_name):
            if verbose:
                print(f'skipped "{file_name}"')

            continue

        if verbose:
            print(f'[{i + 1}/{len(files)}] reading "{file_name}"..')

        path = os.path.join(folder, file_name)
        npz = np.load(path)

        if isinstance(dataset, dict):
            for k in keys:
                dataset[k].append(np.array(npz[k], dtype=dtype))

            count += len(dataset[keys[0]][-1])
        else:
            dataset.append(np.array(npz[keys[0]], dtype=dtype))
            count += len(dataset[-1])

        if count >= limit:
            if verbose:
                print(f'[break] limit of {limit} reached.')
            break

    # finally, stack each image over the batch dimension
    if isinstance(dataset, dict):
        return {k: np.concatenate(v, axis=0) for k, v in dataset.items()}

    return np.concatenate(dataset, axis=0)


def from_h5_to_npz(src: str, dst: str, dtype=np.float16):
    """Reads .h5 files and converts them into .npz"""
    makedir(dst)
    files = os.listdir(src)

    for i, file_name in enumerate(files):
        print(f'[{i + 1}/{len(files)}] reading "{file_name}"..')

        path = os.path.join(src, file_name)

        # each file contains N 286x360 images of the plane (eta, phi)
        with h5py.File(path, 'r') as file:
            # inner-tracker image
            image_trk = np.array(file.get('ImageTrk_PUcorr'), dtype=dtype)

            # ECAL image
            image_ecal = np.array(file.get('ImageECAL'), dtype=dtype)

            # HCAL image
            image_hcal = np.array(file.get('ImageHCAL'), dtype=dtype)

            # stack the three images to form 3-channel images
            # shape: (N, 286, 360, 3)
            images = np.stack([image_trk, image_ecal, image_hcal], axis=-1)

            # transpose to have (phi, eta) instead of (eta, phi)
            # shape: (N, 360, 286, 3)
            images = np.transpose(images, axes=[0, 2, 1, 3])

            # save
            save_path = os.path.join(dst, file_name)
            save_path, _ = os.path.splitext(save_path)  # remove .h5 extension

            np.savez_compressed(save_path, images)
            print(f'  -> saved at "{save_path}.npz"')

        # cleanup
        del file, image_trk, image_ecal, image_hcal, images
        free_mem()


# TODO: broken when there are sub-folders
def load_from_checkpoint(model: tf.keras.Model, path: str, mode: str, base_dir='weights', **kwargs):
    """Load the weights of a pre-built model"""
    path = os.path.join(base_dir, path)

    # list all files in directory
    files = os.listdir(path)

    # split into (path, ext) tuples
    files = [os.path.splitext(os.path.join(path, fname)) for fname in files]

    # keep only weights files
    files = filter(lambda x: 'data-' in x[1], files)

    # from tuples get only path; remove ext
    files = map(lambda x: x[0], files)

    # zip files with metric value
    files_and_metric = map(lambda x: (x, x.split('-')[-1]), files)

    # sort by metric value
    files = sorted(files_and_metric, key=lambda x: x[-1], reverse=mode.lower() == 'min')
    files = map(lambda x: x[0], files)
    files = list(files)

    # load the best weights
    print(f'Loaded from "{files[-1]}"')
    model.load_weights(files[-1], **kwargs)


def get_checkpoint(path: str, monitor: str, mode: str, best_only=True):
    path = os.path.join('weights', path, 'weights-{epoch:02d}-' + f'\u007b{monitor}:.3f\u007d')

    return ModelCheckpoint(path,
                           save_weights_only=True, monitor=monitor,
                           mode=mode, save_best_only=bool(best_only))


def get_tensorboard(folder: str, **kwargs):
    logdir = f"logs/{folder}/" + actual_datetime()
    return tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                          write_graph=False, **kwargs)


def actual_datetime() -> str:
    """Returns the current data timestamp, formatted as follows: YearMonthDay-HourMinuteSecond"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def inspect_model(base_model: tf.keras.Model, layer_class) -> tf.keras.Model:
    # https://androidkt.com/get-output-of-intermediate-layers-keras/
    outputs = []

    for layer in base_model.layers:
        if isinstance(layer, layer_class):
            outputs.append(layer.output)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs)
