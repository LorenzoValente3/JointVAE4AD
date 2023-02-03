
import numpy as np


# keys of each `.h5` file
KEYS = ['ImageECAL', 'ImageECAL_noPU', 'ImageHCAL',
        'ImageHCAL_noPU', 'ImageTrk', 'ImageTrk_PUcorr',
        'ImageTrk_noPU']

STR_TO_CLASS = {'qcd': 0, 'h125': 1, 'h400': 2, 'h700': 3, 'h1000': 4}

# maximum value (of train-set) of each image channel
MAX_VAL = np.array([6770.0, 2192.0, 115.56])

# minimum value (excluding non-zero pixels) computed on normalized images
MIN_VAL = np.array([1.043e-5, 0.0002031, 0.000516])


MIN_ENERGY = np.array([0.08, 0.4, 0.05], dtype=np.float32)
MAX_CLIPPED_ENERGY = np.array([250.0, 100.0, 100.0], dtype=np.float32)


BKG_INDEX = 0
LABELS = ['QCD', 'SUEP', 'SVJ']
MASSES = {1: {1: 125, 2: 400, 3: 700, 4: 1000},  # SUEP (GeV)
          2: {5: 2.1, 6: 3.1, 7: 4.1}}  # SVJ (TeV)


def set_labels(labels: list, bkg_index: int):
    global LABELS, BKG_INDEX
    assert len(labels) > 0

    LABELS = labels
    BKG_INDEX = int(bkg_index)


def set_masses(masses: dict):
    global MASSES
    MASSES = masses
