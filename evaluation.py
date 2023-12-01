import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import time

import sklearn.model_selection

from tensorflow import keras as tfk
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import schedules
from tensorflow_addons.optimizers import AdamW

# from qkeras import *
# from qkeras.quantizers import *

import ad

from ad import utils
from ad.models.jointvae import JointVAE
from ad.models.qjointvae import QJointVAE

from typing import List


ad.constants.set_labels(['qcd', 'top'], bkg_index=0)
ad.constants.set_masses(masses={i: {i: i} for i in range(2)})

utils.set_random_seed(42)
ad.plot.set_style()

# Load Data

# load all the data (both classes)
data = np.load('/beegfs/desy/user/valentel/JointVAE4AD/data/qcd_vs_top_pt.npz')

x = data['images']
y = data['labels']

print(x.shape)
# divide data per-class
qcd = {k: v[y == 0] for k, v in data.items()}  # <-- train on this
top = {k: v[y == 1] for k, v in data.items()}

x_train, x_valid = sklearn.model_selection.train_test_split(qcd['images'], train_size = 0.75 )
y_train, y_valid = sklearn.model_selection.train_test_split(qcd['labels'], train_size = 0.75)

######################
#######  MODEL  ######
######################

# model = JointVAE(
#                       name = 'JointVAE',
#                       continous_latent = 32,
#                       discrete_latent= 20,
#                       temperature = 50,
#                       alpha = 1.,
#                       beta = 3e3,
#                       eps_kl = 1e-5,

#                       encoder=dict(depths=(1, 1, 1, 1),
#                                    filters=[32, 64, 128, 128],
#                                    activation = tf.nn.leaky_relu ,
#                                    kernel_initializer='he_normal',
#                                    groups=4,
#                                    input_shape=(40, 40, 1)),
#                       decoder=dict(depths=(1, 1, 1, 1),
#                                    filters=[32, 128, 128, 64],
#                                    activation = tf.nn.leaky_relu ,
#                                    kernel_initializer='he_normal',
#                                    groups=8,
#                                    size = (3, 3, 256),
#                                    crop=(40, 40))
#                         )
# model.summary()

from ad.models.qcd_what_ae import HeimelJVAE

model = HeimelJVAE(
                      name = 'HeimelJVAE',
                      image_shape=(40, 40, 1),
                      continous_latent = 32,
                      discrete_latent= 20,
                      temperature = 50,
                      beta = 3e3,
                      eps_kl = 1e-7,

                 encoder=dict(filters=[(10, None), (5, 5)],
                              units=[400, 100], kernel_initializer='he_uniform'),
                 decoder=dict(filters=[(5, 5), (5, 10)],
                              units=[100, 400], reshape_to=(20, 20, 1),
                              crop = (40,40),
                              bias=-1.0, kernel_initializer='he_uniform')

                 )

model.summary()


# Load from Checkpoint
ckpt_path = '/beegfs/desy/user/valentel/JointVAE4AD/weights/jointvae/'
ckpt_weights = 'weights-01-10.600'
model.load_weights(ckpt_path + ckpt_weights)

top_train, top_valid = sklearn.model_selection.train_test_split(top['images'], train_size = 0.75 )
ytop_train, ytop_valid = sklearn.model_selection.train_test_split(top['labels'], train_size = 0.75 )

# Anomaly Detection Plots
scores = ad.evaluation.compute_scores(model, x= np.concatenate((x_valid, top_valid),axis=0), batch_size=128)#256)
scores = {k: v.reshape((-1, 1)) for k, v in scores.items()}
scores =ad.evaluation.latents_as_scores(scores, np.concatenate ((y_valid, ytop_valid),axis=0), np.concatenate ((y_valid, ytop_valid),axis=0))

curves = ad.plot.roc_per_mass(bkg_scores=scores['qcd'],
                              signal_scores=scores['top'],
                            #   bins = 8000,
                              fontsize=40,
                            #   x_limits={'bce_1': (0, 150),
                            #             'total_1': (0, 180),
                            #             'kl_cont_1': (0, 7.5),
                            #             'kl_disc_1': (0, .14),
                            #             'kl_tot_1': (0, 7.5), },
                              path = 'results/Heimel/' + ckpt_weights + '/' ,
                              )