#!/usr/bin/env python

import numpy as np
from typing import Dict, List, Any
from scipy.special import softmax, expit
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import pickle
#import tensorflow_probability as tfp
#tfd = tfp.distributions

def load_table_from_pickle(infile: str) -> List[Any]:
    table = pickle.load(open(infile, "rb"))
    bin_info = dict()
    bin_info['dist'] = {'c': table['bin_centers'][0],
            'e': table['bin_edges'][0],
            'w': table['bin_widths'][0]}

    bin_info['rho'] = {'c': table['bin_centers'][1],
            'e': table['bin_edges'][1],
            'w': table['bin_widths'][1]}

    bin_info['z'] = {'c': table['bin_centers'][2],
            'e': table['bin_edges'][2],
            'w': table['bin_widths'][2]}

    bin_info['dt'] = {'c': table['bin_centers'][3],
            'e': table['bin_edges'][3],
            'w': table['bin_widths'][3]}

    return table, bin_info


def load_table_from_pickle_cascade(infile: str) -> List[Any]:
    table = pickle.load(open(infile, "rb"))
    bin_info = dict()
    bin_info['dist'] = {'c': table['bin_centers'][0],
            'e': table['bin_edges'][0],
            'w': table['bin_widths'][0]}

    bin_info['azi'] = {'c': table['bin_centers'][1],
            'e': table['bin_edges'][1],
            'w': table['bin_widths'][1]}

    bin_info['cos_zen'] = {'c': table['bin_centers'][2],
            'e': table['bin_edges'][2],
            'w': table['bin_widths'][2]}

    bin_info['dt'] = {'c': table['bin_centers'][3],
            'e': table['bin_edges'][3],
            'w': table['bin_widths'][3]}

    return table, bin_info


def get_bin_idx(val: float, bins: np.ndarray) -> int:
    assert np.logical_and(val > bins[0], val < bins[-1]), f'value {val} not within bounds [{bins[0]}, {bins[-1]}]'
    return np.digitize(val, bins, right=False)-1


def adjust_plot_1d(fig, ax, plot_args=None):
    if not plot_args:
        plot_args = {}

    for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(1.5)
          ax.spines[axis].set_color('0.0')

    y_scale_in_log = plot_args.get('y_axis_in_log', False)
    if(y_scale_in_log):
        ax.set_yscale('log')

    ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=18)
    ax.yaxis.set_ticks_position('both')
    ax.set_ylabel(plot_args.get('ylabel', 'pdf'), fontsize=20)
    ax.set_xlabel(plot_args.get('xlabel', 'var 1'), fontsize=20)
    ax.set_ylim(plot_args.get('ylim', [0, 1]))
    ax.set_xlim(plot_args.get('xlim', [0, 1]))
    ax.legend()


def get_binomial_photons(dist, azi, cos_zen, table, bin_info, error_scale=1, as_float=False):
    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(azi, bin_info['azi']['e'])
    k = get_bin_idx(cos_zen, bin_info['cos_zen']['e'])

    n_photons_in_each_bin = copy.copy(table['values'][i, j, k, :])
    n_photons_in_each_bin /= error_scale

    total = np.sum(n_photons_in_each_bin)

    if not as_float:
        return np.rint(n_photons_in_each_bin).astype(int), int(np.rint(total))
    else:
        return n_photons_in_each_bin, total

def get_nphotons_vals(dist, azi, cos_zen, table, bin_info):
    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(azi, bin_info['azi']['e'])
    k = get_bin_idx(cos_zen, bin_info['cos_zen']['e'])

    prob_vals = copy.copy(table['values'][i, j, k, :])
    return copy.copy(prob_vals)


def get_effective_binomial_photons(dist, rho, z, table, bin_info, error_scale=1, as_float=False):
    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(rho, bin_info['rho']['e'])
    k = get_bin_idx(z, bin_info['z']['e'])

    prob_vals = copy.copy(table['values'][i, j, k, :])
    prob_err = copy.copy(table['weights'][i, j, k, :])

    if not as_float:
        total_statistical_power = int(np.sum(prob_vals) ** 2 / np.sum(prob_err)) / error_scale
    else:
        total_statistical_power = np.sum(prob_vals) ** 2 / np.sum(prob_err) / error_scale

    fracs_in_each_bin = prob_vals / np.sum(prob_vals)
    n_photons_in_each_bin = fracs_in_each_bin * total_statistical_power

    if not as_float:
        return np.rint(n_photons_in_each_bin).astype(int), int(np.rint(total_statistical_power))
    else:
        return n_photons_in_each_bin, total_statistical_power

def get_effective_binomial_photons_cascade(dist, azi, cos_zen, table, bin_info, error_scale=1, as_float=False):
    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(azi, bin_info['azi']['e'])
    k = get_bin_idx(cos_zen, bin_info['cos_zen']['e'])

    prob_vals = copy.copy(table['values'][i, j, k, :])
    prob_err = copy.copy(table['weights'][i, j, k, :])

    if not as_float:
        total_statistical_power = int(np.sum(prob_vals) ** 2 / np.sum(prob_err)) / error_scale
    else:
        total_statistical_power = np.sum(prob_vals) ** 2 / np.sum(prob_err) / error_scale

    fracs_in_each_bin = prob_vals / np.sum(prob_vals)
    n_photons_in_each_bin = fracs_in_each_bin * total_statistical_power

    if not as_float:
        return np.rint(n_photons_in_each_bin).astype(int), int(np.rint(total_statistical_power))
    else:
        return n_photons_in_each_bin, total_statistical_power


def get_prob_vals(dist, rho, z, table, bin_info):
    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(rho, bin_info['rho']['e'])
    k = get_bin_idx(z, bin_info['z']['e'])

    prob_vals = copy.copy(table['values'][i, j, k, :])
    return copy.copy(prob_vals)


def get_err_vals(dist, rho, z, table, bin_info):
    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(rho, bin_info['rho']['e'])
    k = get_bin_idx(z, bin_info['z']['e'])

    err_vals= copy.copy(table['weights'][i, j, k, :])
    return copy.copy(np.sqrt(err_vals))


def make_dt_plot_w_pdf(dist, rho, z, zenith, azimuth, pars, table, bin_info, peakyness=0.3, scale=10, logscale=False, dtype=tf.float64, outfile="test.png"):
    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(rho, bin_info['rho']['e'])
    k = get_bin_idx(z, bin_info['z']['e'])

    prob_vals = copy.copy(table['values'][i, j, k, :])
    prob_err = copy.copy(np.sqrt(table['weights'][i, j, k, :]))


    tot_prob = np.sum(prob_vals)
    prob_vals /= tot_prob
    prob_vals /= bin_info['dt']['w']

    prob_err /= tot_prob
    prob_err /= bin_info['dt']['w']

    fig, ax = plt.subplots()
    ax.hist(bin_info['dt']['c'], bins=bin_info['dt']['e'], weights=prob_vals, histtype='step', lw=1,
            label=f"(d={dist:.1f}m, $\\rho$ ={rho:.2f} rad, z={z:.0f}m)", color='gray')
    ax.errorbar(bin_info['dt']['c'], prob_vals, yerr = prob_err, lw=0, elinewidth=2,color='k', zorder=100)

    plot_args = {'xlim':[-2, scale*dist],
                 'ylim':[0.0, 1.2 * np.amax(prob_vals)],
                 'xlabel':'dt [ns]',
                 'ylabel':'pdf'}

    xvals = np.linspace(-50, scale*dist, 1000)
    xv = tf.constant(xvals, dtype=dtype)

    w = pars[:3]

    sigma = pars[3:6]
    mu = pars[6:]

    mu_sq = mu * mu
    sigma_sq = sigma*sigma
    g_a = peakyness + mu_sq / sigma_sq
    g_b = mu / sigma_sq

    #print(g_a)

    gm = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          probs=tf.constant(w, dtype=dtype)),
      components_distribution=tfd.Gamma(
        concentration=tf.constant(g_a, dtype=dtype),       # One for each component.
        rate = tf.constant(g_b, dtype=dtype),
        force_probs_to_zero_outside_support=True))

    yvals = gm.prob(xv).numpy()
    plt.plot(xvals, yvals, 'r-', zorder=10, linewidth=2, label='3-comp Gamma')
    for i,c in zip(range(3), ['purple', 'plum', 'pink']):
        tw = w[i]
        tg = tfd.Gamma(concentration=tf.constant(g_a[i], dtype=dtype),       # One for each component.
                        rate = tf.constant(g_b[i], dtype=dtype),
                        force_probs_to_zero_outside_support=True)
        yvals = tw * tg.prob(xv).numpy()
        plt.plot(xvals, yvals, color=c,linestyle='solid', zorder=8, linewidth=1.5,
                 label=f'Gamma {i}')

    adjust_plot_1d(fig, ax, plot_args=plot_args)
    ax.set_title(f"infinite $\mu$ (zenith={np.rad2deg(zenith):.0f}deg, azimuth={np.rad2deg(azimuth):.0f}deg)")
    if logscale:
        plt.yscale('log')
        plt.ylim(ymin=1.e-5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    del fig

def convert_pars(pars, w_zero_peak=False):

    if not w_zero_peak:
        w = softmax(pars.numpy()[:3])
        z1 = 0.1 + tf.math.softplus(pars.numpy()[3:6])
        z2 = 0.1 + tf.math.softplus(pars.numpy()[6:])


        pars = np.concatenate([w, z1, z2])
        pars[6:] = pars[3:6] + pars[6:]

    else:
        w =  np.concatenate([expit(pars.numpy()[:1]), softmax(pars.numpy()[1:4])])
        print(w)
        z1 = tf.math.softplus(pars.numpy()[4:7])
        z2 = tf.math.softplus(pars.numpy()[7:])


        pars = np.concatenate([w, z1, z2])
        pars[7:] = pars[4:7] + pars[7:]
    return pars

def get_seed_dict_distance():
    d = dict()
    d[5] = {'loc':0, 'weights': softmax([np.log(0.3), np.log(0.2), np.log(0.1)]), 'sigma': np.array([1, 20, 100]), 'ds': np.array([2, 15, 90])}
    #d[5] = {'loc':1, 'weights': softmax([np.log(0.9), np.log(0.01), np.log(0.001)]), 'sigma': np.array([3, 35, 500]), 'ds': np.array([1, 5, 50])}
    d[7] = {'loc':1, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.03)]), 'sigma': np.array([0.75, 2, 100]), 'ds': np.array([0.3, 4, 20])}
    #d[7] = {'loc':1, 'weights': softmax([np.log(0.9), np.log(0.01), np.log(0.001)]), 'sigma': np.array([3, 35, 500]), 'ds': np.array([1, 5, 50])}
    d[12] = {'loc':1, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([8, 30, 300]), 'ds': np.array([4, 20, 50])}
    d[19] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([15, 40, 300]), 'ds': np.array([4, 25, 50])}
    d[34] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([25, 45, 100]), 'ds': np.array([25, 44, 200])}
    #d[34] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([25, 100, 300]), 'ds': np.array([25, 35, 50])}
    d[57] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([60, 150, 300]), 'ds': np.array([60, 120, 300])}
    d[99] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([100, 200, 400]), 'ds': np.array([150, 200, 350])}
    d[153] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([150, 400, 600]), 'ds': np.array([150, 400, 600])}
    d[191] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([150, 500, 900]), 'ds': np.array([200, 500, 900])}
    d[296] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([300, 650, 1200]), 'ds': np.array([200, 650, 1200])}
    d[366] = {'loc':2, 'weights': softmax([np.log(0.3), np.log(0.3), np.log(0.3)]), 'sigma': np.array([300, 900, 1500]), 'ds': np.array([300, 900, 1500])}
    return d
