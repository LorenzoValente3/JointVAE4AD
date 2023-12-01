# Import the necessary libraries
from comet_ml import Experiment
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow import keras as tfk
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

# Import your custom library (ad)
import ad
from ad import utils
from ad.models.jointvae import JointVAE

# Import additional dependencies
from typing import List
import sklearn.model_selection

# Load your Comet API Key from a file
with open('./ad/comet_api_key.txt', 'r') as file:
    key = file.read()

# Initialize a Comet experiment
experiment = Experiment(
    api_key=key,
    project_name="jvae",
    workspace="lorenzovalente3",
    # display_name="jave_first"
)

# Set class labels and masses
ad.constants.set_labels(['qcd', 'top'], bkg_index=0)
ad.constants.set_masses(masses={i: {i: i} for i in range(2)})

# Set a random seed and style for plotting
utils.set_random_seed(42)
ad.plot.set_style()


# Load your data
data = np.load('./data/qcd_vs_top_pt.npz')
x = data['images']
y = data['labels']

# divide data per-class
qcd = {k: v[y == 0] for k, v in data.items()}  # <-- train on this
top = {k: v[y == 1] for k, v in data.items()}

# Split the data into training and validation sets
x_train, x_valid = sklearn.model_selection.train_test_split(qcd['images'], train_size=0.75)
# y_train, y_valid = sklearn.model_selection.train_test_split(qcd['labels'], train_size=0.75)

# Model
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


# Compile your model with the Adam optimizer
model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1))

# Define callbacks for training
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=True, patience=4, restore_best_weights=True)

# Assuming the utils.get_checkpoint function is defined elsewhere in your code.
checkpoint = utils.get_checkpoint(path='qcdorwhat_first', monitor='val_loss', mode='min', best_only=False)
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Training your model
start_loop = time.time()
h = model.fit(x=x_train, validation_data=(x_valid,), verbose=1, epochs=15, batch_size=128,
              callbacks=[early_stop, checkpoint, reduceLR])

# Print the total time taken for training
print("\nTotal time taken: %.2fmin" % ((time.time() - start_loop) / 60.))

# Log additional metrics
for epoch in range(len(h.history['loss'])):
    # Log training loss
    experiment.log_metric("train_loss", h.history['loss'][epoch], step=epoch)
    # Log validation loss
    experiment.log_metric("val_loss", h.history['val_loss'][epoch], step=epoch)

# End the experiment
experiment.end()
