
import os
import numpy as np
import tensorflow as tf

import ad
from ad import utils
from ad.models import ConvAE


def get_valid_set(clip_fn, base_path='data/supervised-k5-plain', folder='valid') -> tuple:
    x_valid = utils.read_npz(folder=os.path.join(base_path, folder), dtype=np.float32,
                             verbose=False,
                             shuffle=False, keys=['images', 'labels', 'masses'])

    y_valid = x_valid['labels']
    m_valid = x_valid['masses']

    if clip_fn is None:
        x_valid = x_valid['images']
    else:
        assert callable(clip_fn)
        x_valid = clip_fn(x_valid['images'])

    x_mask = (x_valid > 0.0).astype(np.float32)
    utils.free_mem()

    return x_valid, y_valid, m_valid, x_mask


def get_test_set(clip_fn, base_path='data/supervised-k5-plain') -> tuple:
    # NOTE: in the unsupervised setting, the training split of SUEP is NOT used!
    x_suep = utils.read_npz(folder=os.path.join(base_path, 'train'), dtype=np.float32,
                            verbose=False, shuffle=False, skip='qcd',
                            keys=['images', 'labels', 'masses'])

    x_qcd = utils.read_npz(folder=os.path.join(base_path, 'test'), dtype=np.float32,
                           verbose=False, shuffle=False,
                           keys=['images', 'labels', 'masses'])

    if clip_fn is None:
        x_test = np.concatenate([x_suep['images'], x_qcd['images']])
    else:
        assert callable(clip_fn)
        x_test = np.concatenate([clip_fn(x_suep['images']), clip_fn(x_qcd['images'])])

    y_test = np.concatenate([x_suep['labels'], x_qcd['labels']])
    m_test = np.concatenate([x_suep['masses'], x_qcd['masses']])
    x_mask = (x_test > 0.0).astype(np.float32)

    del x_suep, x_qcd
    utils.free_mem()

    return x_test, y_test, m_test, x_mask


def get_threshold_at(efficiency: float, roc: dict) -> float:
    """Given a `roc` curve, it computes the threshold value to achieve the desired signal `efficiency`"""
    index = np.abs(roc['tpr'] - efficiency).argmin()
    return roc['thresholds'][index]


def latents_as_scores(latents: dict, y: np.ndarray, masses: np.ndarray) -> dict:
    from ad.constants import MASSES, BKG_INDEX
    scores = {}

    # loop over class labels
    for label in np.unique(y):
        mask = y == label
        name = utils.get_name(label).lower()
        score = {}

        # if is a signal
        if label != BKG_INDEX:
            # loop over mass labels, and select only the relevant latents
            for m, mass in utils.get_masses(label).items():
                score_m = {}
                mass_mask = masses == m

                for k, z in latents.items():
                    for i in range(z.shape[-1]):
                        score_m[f'{k}_{i + 1}'] = z[mask & mass_mask][:, i]

                score[mass] = score_m
        else:
            # QCD (or background)
            for k, z in latents.items():
                for i in range(z.shape[-1]):
                    score[f'{k}_{i + 1}'] = z[mask][:, i]

        scores[name] = score

    return scores


def average_predictions(model, x: np.ndarray, m: np.ndarray, plot_args=None, **kwargs):
    from ad.constants import MASSES

    if not isinstance(plot_args, dict):
        plot_args = dict(cmap=ad.plot.CMAP2, v_max=None)

    y = model.predict(x, **kwargs)

    for mass in np.unique(m):
        # find mass name
        mass_key = None

        for k, v in MASSES.items():
            if mass in v:
                mass_key = k
                break

        if mass_key is None:
            print('QCD')
        else:
            print(f'{utils.get_name(label=mass_key)} ({utils.get_mass(mass_key, mass)})')

        mask = m == mass

        x_mu = np.mean(x[mask], axis=0)
        y_mu = np.mean(y[mask], axis=0)

        ad.plot.compare(x_mu, y_mu, **plot_args)


# ------------------------------------------------------------------------------------------------
# -- Anomaly Detection
# -------------------------------------------------------------------------------------------------

def cross_entropy_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    loss = tf.keras.losses.binary_crossentropy(true, pred, axis=[])
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
    return loss.numpy()


def dice_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    sum_p = tf.reduce_sum(tf.square(pred), axis=[1, 2, 3])
    sum_t = tf.reduce_sum(tf.square(true), axis=[1, 2, 3])

    union = sum_p + sum_t
    intersection = tf.reduce_sum(pred * true, axis=[1, 2, 3])

    loss = (union / (2 * intersection)).numpy()
    loss[np.isinf(loss) | np.isnan(loss)] = 0.0
    return loss


def mse_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    loss = tf.square(true - pred)
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
    return loss.numpy()


def energy_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    pred_energy = tf.reduce_sum(pred, axis=[1, 2, 3])
    true_energy = tf.reduce_sum(true, axis=[1, 2, 3])

    loss = []
    for p, t in zip(pred_energy, true_energy):
        loss.append(tf.keras.losses.huber(p[tf.newaxis], t[tf.newaxis]))

    return tf.stack(loss).numpy()


def diff_energy(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    pred_energy = tf.reduce_sum(pred, axis=[1, 2, 3])
    true_energy = tf.reduce_sum(true, axis=[1, 2, 3])

    return (true_energy - pred_energy).numpy()


def compute_scores(model, x: np.ndarray, batch_size=128 ):
    """Scores computation for the jointVAE model"""
    scores = dict(energy_diff=[], bce=[], dice=[], total=[], 
                  mse=[], kl_cont=[], kl_disc=[], kl_tot=[])

    for batch in tf.data.Dataset.from_tensor_slices(x).batch(batch_size):
        q, z_mean, z_log_var = model.encoder(batch)
        # print(q.shape)
        # break

        z = model.sampling([z_mean, z_log_var, q])
        y = model.decoder(z)
        
        # scores
        energy = tf.reduce_sum(batch, axis=[1, 2, 3]) - tf.reduce_sum(y, axis=[1, 2, 3])

        bce = cross_entropy_loss(y, batch)
        dice = dice_loss(y, batch)
        total = bce + dice
        mse = mse_loss(y, batch)
        
        ##### kl_continous #####
        kl_cont_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_cont_loss = tf.reduce_sum(kl_cont_loss, axis=1)
        kl_cont = model.beta * kl_cont_loss    

        ##### kl_categorical #####
        q_p = tf.nn.softmax(q, axis=-1) # Convert the categorical codes into probabilities
        # Entropy of the logits
        h1 = q_p * tf.math.log(q_p + model.eps_kl)
        # Cross entropy with the categorical distribution
        h2 = q_p * tf.math.log(1. / model.discrete_latent + model.eps_kl)
        kl_disc = tf.reduce_sum(h1 - h2 , axis = 1 ) * model.beta

        kl_tot = kl_cont + kl_disc
        
        # append scores                 
        for k, v in zip(scores.keys(), [bce, dice, energy, total,mse ,kl_cont, kl_disc, kl_tot]):
            scores[k].append(v)

    return {k: np.concatenate(v) for k, v in scores.items()}
    # return scores

def hls_array(filename :str):
    q = []
    mean = []
    log_var = []

    file = open(filename, 'r')
    lines = file.readlines()
    len(lines)

    def convert_line(line: str) -> np.ndarray:
        numbers = line.split()
        array = np.array([float(n) for n in numbers])
        return array.reshape((1, -1))

    # read lines in chunks of three
    for i in range(0, len(lines), 3):
        q.append(convert_line(line=lines[i]))
        mean.append(convert_line(line=lines[i + 1]))
        log_var.append(convert_line(line=lines[i + 2]))

    # concat everything
    q = np.concatenate(q)
    mean = np.concatenate(mean)
    log_var = np.concatenate(log_var)

    return q, mean, log_var


def compute_hls_scores(model, q: np.ndarray, mean: np.ndarray, var: np.ndarray):
    """Scores computation for the jointVAE model"""
    scores = dict(kl_cont=[], kl_disc=[], kl_tot=[])
    
    for i in range(q.shape[0]):
        q_i, z_mean_i, z_log_var_i = q[i], mean[i], var[i]
               
        ##### kl_continous #####
        kl_cont_loss = -0.5 * (1 + z_log_var_i - tf.square(z_mean_i) - tf.exp(z_log_var_i))
        kl_cont_loss = tf.reduce_sum(kl_cont_loss, axis=0)
        kl_cont = model.beta * kl_cont_loss    

        ##### kl_categorical #####
        q_p = tf.nn.softmax(q_i, axis=0) # Convert the categorical codes into probabilities
        # Entropy of the logits
        h1 = q_p * tf.math.log(q_p + model.eps_kl)
        # Cross entropy with the categorical distribution
        h2 = q_p * tf.math.log(1. / model.discrete_latent + model.eps_kl)
        kl_disc = tf.reduce_sum(h1 - h2 , axis = 0 ) * model.beta
        kl_tot = kl_cont + kl_disc

        # append scores                 
        for k, v in zip(scores.keys(), [kl_cont, kl_disc, kl_tot]):
            scores[k].append(v)

    return {k: np.concatenate(v) for k, v in scores.items()}


def anomaly_scores(model, x: np.ndarray, y: np.ndarray, m: np.ndarray, batch_size=128, **kwargs) -> dict:
    from ad.constants import MASSES
    scores = {}
    
    for label in np.unique(y):
        name = utils.get_name(label).lower()
        mask = y == label

        if label in MASSES:
            score = {}
            for m_, mass in utils.get_masses(label).items():
                score[mass] = compute_scores(model, x=x[mask & (m == m_)], batch_size=batch_size)
        else:
            # QCD
            score = compute_scores(model, x=x[mask], batch_size=batch_size)

        scores[name] = score

    return scores


def compute_roc(bkg_scores: dict, signal_scores: dict) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = dict()

    for k, bkg_score in bkg_scores.items():
        curves[k] = {}

        for h, score in signal_scores.items():
            sig_score = score[k]
            key = f'{utils.get_name_from(mass=h)} ({h})'

            # compute roc
            y_true = np.concatenate([
                np.zeros_like(bkg_score), np.ones_like(sig_score)])

            y_score = np.concatenate([bkg_score, sig_score])

            fpr, tpr, t = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)

            curves[k][key] = dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=t)

    return curves