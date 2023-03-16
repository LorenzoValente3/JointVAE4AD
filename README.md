# JointVAE for Anomaly Detection

Anomaly Detection of HEP data done with Joint Variational Autoencoder.

The aim is to discriminate the top jets signals from the QCD background, but in an *unsupervised setting* (i.e. we only train on the QCD samples).
The data being analyzed has been sourced from the [git repo](https://github.com/bmdillon/jet-preprocessing).

## Project Structure

* [`ad`](./ad): the main namespace.
  *  [`models`](./ad/models): defines custom models.
  *  [`layers`](./ad/layers.py): defines custom layers used by the models.
  *  [`metrics`](./ad/metrics.py): custom metrics and wrappers.
  *  [`evaluation`](./ad/evaluation.py): code for evaluating the models.
  *  [`utils`](./ad/utils.py): general utility code.
  *  [`plot`](./ad/plot.py): plot utils.
  *  [`aug`](./ad/aug.py): data augmentations.
*  [`weights`](./weights): contains the pre-trained weights of the various models.
