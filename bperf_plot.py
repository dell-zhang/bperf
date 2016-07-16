from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
import bperf_stats as bps

__all__ = ['plot_post', 'plot_trace', 'plot_multi_trace']

def plot_post(sample, cred_mass=0.95, cmp_val=None, rope=None, 
              xlab='parameter', ylab=None, xlim=None, ylim=None, 
              fontsize=9, labelsize=9, title='', framealpha=0.5, 
              facecolor='skyblue', edgecolor='white', 
              show_mode=False, bins=50):
    post = bps.post_summary(sample, cred_mass, cmp_val, rope)
    # Plot histogram.
    n, bins, patches = plt.hist(sample, normed=True, bins=bins,
                                edgecolor=edgecolor, facecolor=facecolor)
    if xlab:
        plt.xlabel(xlab, fontsize=labelsize)
    if ylab:
        plt.ylabel(ylab, fontsize=labelsize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.title(title, fontsize=fontsize)
    # Display mean or mode:
    if show_mode:
        plt.plot(0, label='mode = %+0.3f' % post['mode'], alpha=0)
    else:
        plt.plot(0, label='mean = %+0.3f' % post['mean'], alpha=0)
    #
    #cv_ht = 0.75 * np.max(n)
    #cen_tend_ht = 0.9 * cv_ht
    cv_ht = 0.75 * (ylim[1] - ylim[0])
    rope_ht = 0.55 * cv_ht
    # Display the comparison value.
    if cmp_val is not None:
        plt.plot([cmp_val, cmp_val], [0, cv_ht], color='darkgreen',
                 linestyle='--', linewidth=2,
                 label=bps.pct_cmp_val_str(cmp_val, post['pct_cmp_val']))
    # Display the ROPE.
    if rope is not None:
        plt.plot([rope[0], rope[0]], [0, 0.96*rope_ht], color='darkred',
                linestyle=':', linewidth=4,
                label="{:5.1%} in ROPE".format(post['pct_in_rope']))
        plt.plot([rope[1], rope[1]], [0, 0.96*rope_ht], color='darkred',
                linestyle=':', linewidth=4)
    # Display the HDI
    plt.plot(post['hdi'], [0, 0], linewidth=6, color='k', 
             label='HDI %.0f%% %s' % (cred_mass*100, bps.interval_str(post['hdi'])))
    plt.legend(loc='upper left', fontsize=labelsize, framealpha=framealpha)

def plot_bfsd(prio_sample, post_sample, cmp_val=0, 
            xlab='parameter', ylab=None, xlim=None, ylim=None,
            labelsize=9, framealpha=0.5):
    prio_pdf_kde = stats.kde.gaussian_kde(prio_sample)
    post_pdf_kde = stats.kde.gaussian_kde(post_sample)
    x = np.linspace(xlim[0], xlim[1], num=100)
    prio_y = prio_pdf_kde(x)
    post_y = post_pdf_kde(x)
    plt.plot(x, prio_y, 'y', linestyle=':', linewidth=4, label='prior')
    plt.plot(x, post_y, 'b', linestyle='-', linewidth=1, label='posterior')
    if xlab:
        plt.xlabel(xlab, fontsize=labelsize)
    if ylab:
        plt.ylabel(ylab, fontsize=labelsize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    cv_ht = 0.75*(ylim[1]-ylim[0])
    bf = post_pdf_kde(cmp_val)/prio_pdf_kde(cmp_val)
    if bf >= 1:
        bf_label = "BF$_{SD}$ = %5.1f" % bf
    else:
        bf_label = "BF$_{SD}$ = %5.3f" % bf
    plt.plot([cmp_val, cmp_val], [0, cv_ht], color='darkgreen',
             linestyle='--', linewidth=2, 
             label=bf_label)
    y0 = prio_pdf_kde(cmp_val)[0]
    y1 = post_pdf_kde(cmp_val)[0] 
    plt.plot(cmp_val, y0, 'yo', markersize=8)
    plt.plot(cmp_val, y1, 'bs', markersize=8)
    #plt.arrow(cmp_val, y0, 0, y1-y0, color='darkgreen', 
    #          width=0.0005, head_width=0.005, head_length=3,
    #          length_includes_head=True)
    plt.legend(loc='upper left', fontsize=labelsize, framealpha=framealpha)

def plot_trace(sample, cmp_val=None, xlab='iteration', ylab='sample value', ylim=None):
    plt.plot(sample, color='skyblue')
    if cmp_val is not None:
        plt.axhline(y=cmp_val, color='darkgreen', linestyle='--', linewidth=2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if ylim:
        plt.ylim(ylim)

def make_2d(a):
    """Ravel the dimensions after the first.
    """
    a = np.atleast_2d(a.T).T
    #flatten out dimensions beyond the first
    n = a.shape[0]
    newshape = np.product(a.shape[1:]).astype(int)
    a = a.reshape((n, newshape), order='F')
    return a

def plot_multi_trace(trace, vars=None,  xlab='iteration', ylabs=None, ylim=None, 
                     figsize=None, lines=None, combined=False):
    """Plot samples histograms and values

    Parameters
    ----------
    trace : result of MCMC run
    vars : list of variable names
        Variables to be plotted, if None all variable are plotted
    ylabs : list of y-axis labels
        The y-axis labels, if None the variable names are displayed
    ylim :  (ymin, ymax)
        The y-limits of each plot
    figsize : figure size tuple
        If None, size is (6, num of variables * 3) inch
    lines : dict
        Dictionary of variable name / value  to be overplotted as vertical
        lines to the posteriors and horizontal lines on sample values
        e.g. mean of posteriors, true values of a simulation
    combined : bool
        Flag for combining multiple chains into a single chain. If False
        (default), chains will be plotted separately.

	Returns
    -------
    fig : figure object
    """
    if vars is None:
        vars = trace.varnames
    if ylabs is None:
        ylabs = trace.varnames
    #
    n = len(vars)
    #
    fig, ax = plt.subplots(n, sharex=True, squeeze=False)
    #
    for i, v in enumerate(vars):
        for d in trace.get_values(v, combine=combined, squeeze=False):
            d = np.squeeze(d)
            d = make_2d(d)
            ax[i, 0].plot(d, color='skyblue')
            ax[i, 0].set_xlabel(xlab)
            ax[i, 0].set_ylabel(ylabs[i])
            if ylim:
                ax[i, 0].set_ylim(ylim)
            if lines:
                try:
                    ax[i, 0].axhline(y=lines[v], color='darkgreen', 
                                     linestyle='--', linewidth=2)
                except KeyError:
                    pass
    #plt.tight_layout()
    #return fig
