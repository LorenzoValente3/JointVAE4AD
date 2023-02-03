import random
import numpy as np
import tensorflow as tf

from ad import plot, utils


class PlotCallback(tf.keras.callbacks.Callback):
    """Custom callback that plots reconstructions"""

    def __init__(self, freq=1, amount=3, **kwargs):
        self.freq = int(freq)
        self.amount = int(amount)
        self.kwargs = kwargs

        self.random_batch = None
        self.num_batches = 0
        self.should_count_batches = True

    def plot(self):
        assert self.random_batch is not None
        batch = self.random_batch
        
        if isinstance(batch, tuple):
            x_true = batch[0]
        else:
            x_true = batch
        
        # take a random `amount` of samples in batch
        x_true = np.random.choice(x_true, size=min(self.amount, len(x_true)),
                                  replace=False)
        
        if len(x_true) == 1:
            x_true = tf.expand_dims(x_true, axis=0)
    
        x_pred = self.model(x_true)

        for x, y in zip(x_true, x_pred):
            plot.compare(x, y, **self.kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            self.plot()
            
    def on_test_batch_begin(self, batch, logs=None):
        if self.random_batch is None:
            # if the batch is not yet selected, pick the first one
            self.random_batch = batch
            
        if self.should_count_batches:
            self.num_batches += 1
        else:
            # take a random batch with prob. 1/n
            if random.random() > 1.0 / self.num_batches:
                self.random_batch = batch
    
    def on_test_end(self, logs=None):
        self.should_count_batches = False