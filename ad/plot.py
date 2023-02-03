import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from ad import utils


CMAP1 = 'RdGy_r'
CMAP2 = 'gist_heat_r'
CMAP3 = 'Greys_r'
DEFAULT_CMAP = None

def doWeights(model):
    """Function for plotting the weight distributions"""
    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[0].numpy().flatten()  
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])
    
    fig = plt.figure(figsize=(10,10))
    
    # weights = [weight.numpy().flatten() for layer in model.layers for weight in layer.weights]
    # weights = np.concatenate(weights)
    # bins = np.linspace(np.min(weights), np.max(weights), 100)

    bins = np.linspace(-.15, .15, 100)
    histosW = np.array(histosW, dtype='object')
    plt.hist(histosW,bins,histtype='stepfilled',stacked=True,label=labelsW, edgecolor='black')
    plt.legend(frameon=False,loc='upper right', fontsize = 'xx-small')
    plt.ylabel('Number of Weights')
    plt.xlabel('Weights')
    plt.figtext(0.2, 0.38,model._name, wrap=True, horizontalalignment='left',verticalalignment='center')


def set_style(default_cmap=CMAP2, **kwargs):

    # further customization
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 20
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.alpha'] = 0.65
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams['figure.figsize'] = (12, 10)
    mpl.rcParams['figure.autolayout'] = False

    for k, v in kwargs.items():
        mpl.rcParams[k] = v
    
    if default_cmap is not None:
        global DEFAULT_CMAP
        DEFAULT_CMAP = default_cmap


def sort_legend(ax, by_value: list, reverse=False) -> tuple:
    """Based on:  https://stackoverflow.com/a/27512450"""
    handles, labels = ax.get_legend_handles_labels()
    _, handles, labels = zip(*sorted(zip(by_value, handles, labels),
                                     key=lambda t: t[0], reverse=bool(reverse)))
    return handles, labels


def get_colormap(which='viridis', bkg_color='white', levels=1024):
    return cm.get_cmap(which, levels).with_extremes(under=bkg_color)


def heatmap(arr: np.array, size=(12, 10), ax=None, show=True, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    sns.heatmap(arr, **kwargs, ax=ax)

    if show:
        plt.show()

    return ax


def track(image: np.ndarray, size=(12, 12), ax=None, save=None, path='plot', **kwargs):
    track = tf.unstack(image, axis=-1)
    track = np.concatenate(track, axis=1)

    ax = heatmap(track, size=size, ax=ax, **kwargs)

    ax.set_title('Trk - ECAL - HCAL', fontsize=15)
    ax.set_ylabel(r'$\phi$ cell', fontsize=15)
    ax.set_xlabel(r'$\eta$ cell', fontsize=15)

    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')

    plt.show()


def plot_track(x: np.ndarray, **kwargs):
    return heatmap(np.sum(x, axis=-1), **kwargs)


def compare(*args, cmap=None, save: str = None, title: list = None,
            path='plot', v_min=0.0, v_max=0.1, **kwargs):
    assert len(args) > 0
    
    cmap = cmap or DEFAULT_CMAP
    axes = utils.get_plot_axes(rows=1, cols=len(args))
    
    if title is None:
        title = ['Ground-truth']

        if len(args) >= 2:
            title.append('Reconstruction')

        if len(args) >= 3:
            title.append('Diff: |GT - Pred|')

        assert len(args) == len(title)
    else:
        assert isinstance(title, list)
        assert len(args) == len(title)

    if isinstance(args[0], list) or len(args[0].shape) == 4:
        i = np.random.choice(len(args[0]))
        print(f'i: {i}')

        args = [x[i] for x in args]
    else:
        i = 0

    for x, ax, text in zip(args, axes, title):
        if v_max == 'none':
            v_max = np.max(x)

        if v_min == 'none':
            v_min = np.min(x)

        plot_track(x, cmap=cmap, ax=ax, vmin=v_min, vmax=v_max, show=False, **kwargs)

        e_t_max = np.max(x).item()
        e_t_tot = np.sum(x).item()

        ax.set_title(f'{text}\n max E = {round(e_t_max, 2)}; total E = {round(e_t_tot, 2)}')
        ax.set_xlabel(r'$\eta$ cell')
        ax.set_ylabel(r'$\phi$ cell')

    plt.tight_layout()

    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}-{i}.png'), bbox_inches='tight')

    plt.show()


def compare_channels(*images, cmap=None, v_min=None, v_max=None, **kwargs):
    text = ['Trk', 'ECAL', 'HCAL']
    cmap = cmap or DEFAULT_CMAP
    
    if isinstance(v_min, (int, float)) or (v_min is None):
        v_min = [v_min] * images[0].shape[-1]
    else:
        assert len(v_min) == images[0].shape[-1]

    if isinstance(v_max, (int, float)) or (v_max is None):
        v_max = [v_max] * images[0].shape[-1]
    else:
        assert len(v_max) == images[0].shape[-1]

    for h in images:
        axes = utils.get_plot_axes(rows=1, cols=3)

        for c, (title, ax) in enumerate(zip(text, axes)):
            ax = heatmap(h[..., c], ax=ax, show=False, cmap=cmap, vmin=v_min[c], vmax=v_max[c], **kwargs)
            ax.set_title(title)

        plt.show()


def energy_histogram(data: dict, x_label: str, ax=None, bins=100, show=True, size=(12, 10),
                     var_range=None, log_scale=False, hatch_fn=None, legend='upper right'):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    if hatch_fn is None:
        hatch_fn = lambda key: 'QCD' in key
    else:
        assert callable(hatch_fn)

    if not isinstance(var_range, tuple):
        x_min = np.inf
        x_max = -np.inf

        for value in data.values():
            if isinstance(value, tuple):
                v, _, _ = value  # (true_e - pred_e, mean, std)
            else:
                v = value

            x_min = min(x_min, v.min())
            x_max = max(x_max, v.max())

        var_range = (x_min, x_max)

    for k, value in data.items():
        if isinstance(value, tuple):
            v, mu, std = value
            label = f'{k}: {round(mu, 2)} ({round(std, 2)})'
        else:
            v = value
            label = k

        ax.hist(v, bins=bins, label=label, range=var_range, density=True,
                histtype='step', hatch='//' if hatch_fn(k) else None)

    ax.set_xlabel(str(x_label))
    ax.set_ylabel('Probability')
    ax.legend(loc=str(legend))

    if log_scale:
        ax.set_yscale('log')

    if show:
        plt.tight_layout()
        plt.show()


def roc_losses(qcd_losses: dict, suep_losses: dict, scale='log', bins=100):
    """Plots a ROC curve using various losses as discriminator"""
    from sklearn.metrics import roc_auc_score, roc_curve

    for k, qcd_loss in qcd_losses.items():
        suep_loss = [losses[k] for _, losses in suep_losses.items()]
        suep_loss = np.concatenate(suep_loss)

        # compute roc
        y_true = np.concatenate([
            np.zeros_like(qcd_loss), np.ones_like(suep_loss)])

        y_score = np.concatenate([qcd_loss, suep_loss])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        # plot
        ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)
        loss_range = (min(qcd_loss.min(), suep_loss.min()),
                      max(qcd_loss.max(), suep_loss.max()))

        # histogram
        ax1.hist(qcd_loss, bins=bins, range=loss_range, density=True,
                 label='QCD', histtype='step')

        ax1.hist(suep_loss, bins=bins, range=loss_range, density=True,
                 label='SUEP', histtype='step', hatch='//')
        ax1.set_xlabel(k)
        ax1.set_ylabel('Probability')
        ax1.legend(loc='upper right')

        # ROC

        ax2.plot(fpr, tpr, label=f'{k}, AUC = {round(auc * 100, 2)}%')

        ax2.set_xlabel('Background efficiency (FPR)')
        ax2.set_yscale(scale)
        ax2.set_ylabel('Signal efficiency (TPR)')
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()


def roc_per_mass(bkg_scores: dict, signal_scores: dict, scale='linear', bins=100,
                 legend_hist='upper right', legend_roc='lower right', fontsize=18):
    """Plots a ROC curve using various losses as discriminator"""
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = dict()

    for k, bkg_score in bkg_scores.items():
        other_score = [scores[k] for _, scores in signal_scores.items()]
        other_score = np.concatenate(other_score)

        # plot
        ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)
        loss_range = (min(bkg_score.min(), other_score.min()),
                      max(bkg_score.max(), other_score.max()))

        # histogram
        ax1.hist(bkg_score, bins=bins, range=loss_range, density=True,
                 label=utils.get_bkg_name(), histtype='step', hatch='//')

        curves[k] = {}

        for h, other_score in signal_scores.items():
            sig_score = other_score[k]
            label = f'{utils.get_name_from(mass=h)} ({h})'

            # compute roc
            y_true = np.concatenate([
                np.zeros_like(bkg_score), np.ones_like(sig_score)])

            y_score = np.concatenate([bkg_score, sig_score])

            fpr, tpr, t = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            curves[k][label] = dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=t)

            ax1.hist(sig_score, bins=bins, range=loss_range, density=True,
                     label=label, histtype='step')
            # ROC
            ax2.plot(fpr, tpr, label=f'{k} ({h}), AUC = {round(auc * 100, 2)}%')

        # if k=='kl_cont':
        #   ax1.set_xlim(0, 20)
        ax1.set_xlabel(k)
        ax1.set_ylabel('Probability', fontsize=fontsize)
        ax1.legend(loc=str(legend_hist), fontsize=fontsize - 4)

        ax2.set_xlabel('Background efficiency (FPR)', fontsize=fontsize)
        ax2.set_yscale(scale)
        ax2.set_ylabel('Signal efficiency (TPR)', fontsize=fontsize)
        ax2.legend(loc=str(legend_roc), fontsize=fontsize - 2)

        plt.tight_layout()
        plt.show()

    return curves


def roc_per_mass_stacked(bkg_scores: dict, signal_scores: dict, scale='linear', bins=100, fontsize=18,
                         legend_hist='upper right', legend_roc='lower right', weight=False,
                         thresholds: dict = None) -> dict:
    """Plots a ROC curve using various losses as discriminator"""
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = dict()
    thresholds = thresholds or {}

    for k, qcd_loss in bkg_scores.items():
        score = [scores[k] for _, scores in signal_scores.items()]
        score = np.concatenate(score)

        # plot
        ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)
        loss_range = (min(qcd_loss.min(), score.min()),
                      max(qcd_loss.max(), score.max()))

        # histogram
        ax1.hist(qcd_loss, bins=bins, range=loss_range, density=True,
                 label=utils.get_bkg_name(), histtype='step')

        scores = []
        labels = []
        curves[k] = {}

        for h, score in signal_scores.items():
            s_loss = score[k]
            key = f'{utils.get_name_from(mass=h)} ({h})'

            scores.append(s_loss)
            labels.append(key)

            # compute roc
            y_true = np.concatenate([
                np.zeros_like(qcd_loss), np.ones_like(s_loss)])

            if weight:
                w_suep = len(qcd_loss) / len(s_loss)
                w = np.concatenate([
                    np.ones_like(qcd_loss), w_suep * np.ones_like(s_loss)])
            else:
                w = None

            y_score = np.concatenate([qcd_loss, s_loss])

            fpr, tpr, t = roc_curve(y_true, y_score, sample_weight=w)
            auc = roc_auc_score(y_true, y_score)

            # ROC
            ax2.plot(fpr, tpr, label=f'{k} ({h}), AUC = {round(auc * 100, 2)}%')

            if key in thresholds:
                # find index
                idx = np.abs(t - thresholds[key]).argmin()
                ax2.scatter(fpr[idx], tpr[idx])

            curves[k][key] = dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=t)

        ax1.hist(scores, bins=bins, range=loss_range, density=True,
                 label=labels, stacked=True, histtype='step', hatch='//')

        ax1.set_xlabel(k, fontsize=fontsize)
        ax1.set_ylabel('Probability', fontsize=fontsize)
        ax1.legend(loc=str(legend_hist), fontsize=fontsize - 4)

        ax2.set_xlabel('Background efficiency (FPR)', fontsize=fontsize)
        ax2.set_ylabel('Signal efficiency (TPR)', fontsize=fontsize)
        ax2.set_yscale(scale)
        ax2.legend(loc=str(legend_roc), fontsize=fontsize - 2)

        plt.tight_layout()
        plt.show()

    return curves


def pixels_hist(*counts, labels: list, bins=100, size=(12, 10), legend='best',
                ax=None, show=True, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    ax.hist(counts, bins=bins, label=labels, histtype='step', density=True, **kwargs)
    ax.set_xlabel('# Non-zero pixels')
    ax.set_ylabel('Frac. of Images')

    ax.legend(loc=str(legend))

    if show:
        plt.show()


def pixels_hist_stacked(qcd_counts, suep_counts: dict, bins=100, size=(12, 10),
                        legend='best', ax=None, show=True, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    x_range = [qcd_counts.min(), qcd_counts.max()]

    for v in suep_counts.values():
        x_range = [min(v.min(), x_range[0]),
                   max(v.max(), x_range[1])]

    ax.hist(qcd_counts, bins=bins, range=x_range, label='QCD',
            histtype='step', density=True, **kwargs)

    ax.hist(list(suep_counts.values()), bins=bins, range=x_range, density=True,
            label=list(suep_counts.keys()), histtype='step', hatch='//', stacked=True)

    ax.set_xlabel('# Non-zero pixels')
    ax.set_ylabel('Frac. of Images')

    ax.legend(loc=str(legend))

    if show:
        plt.show()


def history(h, keys: list, rows=2, cols=2, size=8):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    axes = np.reshape(axes, newshape=[-1])

    for ax, k in zip(axes, keys):
        ax.plot(h.history[k], label='train')
        ax.plot(h.history[f'val_{k}'], label='valid')

        ax.set_xlabel('epoch', fontsize=20)
        ax.set_ylabel(k, rotation="vertical", fontsize=20)

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        ax.grid(alpha=0.5, linestyle='dashed')
        ax.legend()

    fig.tight_layout()


def latent(x, y, title: str, ax=None, size=(12, 10), show=True):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    for i, label in enumerate(np.unique(y)):
        x_ = x[y == label]
        ax.scatter(x_[:, 0], x_[:, 1], s=10, label=utils.get_name(label), zorder=-i)

    ax.set_xlabel(r'$z_0$')
    ax.set_ylabel(r'$z_1$')
    ax.set_title(title)

    ax.legend()

    if show:
        plt.show()


def latent_kde(x: np.ndarray, y: np.ndarray, ax=None, size=(10, 10)):
    from scipy.stats import gaussian_kde

    def kde_hist(x, y, ax, ax_histx, ax_histy, **kwargs):
        x_ = np.sort(x)
        y_ = np.sort(y)

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y, **kwargs)

        # kde histograms
        kde_x = gaussian_kde(x_)
        kde_y = gaussian_kde(y_)

        ax_histx.plot(x_, kde_x(x_))
        ax_histy.plot(kde_y(y_), y_)

    if ax is None:
        fig = plt.figure(figsize=size)
    else:
        fig = ax.figure

    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    for i, label in enumerate(np.unique(y)):
        x_ = x[y == label]

        kde_hist(x_[:, 0], x_[:, 1], ax, ax_histx, ax_histy, s=10, alpha=0.7,
                 label=utils.get_name(label), zorder=-i)

    ax.set_xlabel(r'$z_0$')
    ax.set_ylabel(r'$z_1$')

    ax.legend()
    plt.show()


def feature_maps(x: np.ndarray, fill=0.0, cmap=None, normalize=True, eps=1e-5, size=(12, 10)):
    """Visualize feature maps of an inspected model"""
    cmap = cmap or DEFAULT_CMAP

    if len(x.shape) == 4:
        x = np.mean(x, axis=0)

    n = np.ceil(np.sqrt(x.shape[-1]))
    n = int(n)

    image = []

    for i in range(n * n):
        if i % n == 0:
            image.append([])

        if i < x.shape[-1]:
            # normalize the channel
            # https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
            c = x[..., i]

            if normalize:
                c = (c - c.min()) / (c.max() - c.min() + eps)
        else:
            c = np.full_like(x[..., 0], fill_value=fill)

        image[-1].append(c)

    # concatenate channels to form the final image
    image = [np.concatenate(z, axis=1) for z in image]
    image = np.concatenate(image, axis=0)

    # plot
    plt.figure(figsize=size)
    plt.grid(False)

    plt.imshow(image, cmap=cmap, interpolation='none')
    plt.show()
