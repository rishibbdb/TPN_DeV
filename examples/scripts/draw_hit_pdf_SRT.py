import sys, os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--path_to_repo", type=str,
                  default="/home/storage/hans/jax_reco_gupta_corrections4/",
                  dest="PATH_TO_REPO",
                  help="directory containing the reco code")

parser.add_argument("-f", "--file_path", type=str,
                  default="/home/fast_storage/i3/22645/ftr/",
                  dest="PATH_TO_INPUT",
                  help="directory containing the event data .ftr files")

parser.add_argument("-mf", "--meta_file", type=str,
                  default="meta_ds_22645_from_0_to_1000_10_to_100TeV.ftr",
                  dest="META_FILE_NAME",
                  help="Name of the .ftr file containing event meta data")

parser.add_argument("-pf", "--pulses_file", type=str,
                  default="pulses_ds_22645_from_0_to_1000_10_to_100TeV.ftr",
                  dest="PULSES_FILE_NAME",
                  help="Name of the .ftr  file containing event pulses data")

parser.add_argument("-g", "--gpu", type=int,
                  default=0,
                  dest="GPU_INDEX",
                  help="which GPU should run the code")

parser.add_argument("-e", "--event_index", type=int,
                  default=0,
                  dest="EVENT_INDEX",
                  help="Which event should be used. Index within input file.")

parser.add_argument("-ns", "--n_splits", type=int,
                  default=50,
                  dest="N_SPLITS",
                  help="split grid into some number of sequential pieces, to avoid GPU memory limitations")

parser.add_argument("-n", "--network", type=str,
                  default="gupta_4comp_reg",
                  dest="NETWORK",
                  help="options are: gupta_4comp_reg, gupta_4comp, gupta_3comp, gamma_3comp, custom")

parser.add_argument("-s", "--seed", type=str,
                    default="spline_mpe",
                    dest="SEED",
                    help="options are: spline_mpe, truth")

parser.add_argument("-c", "--gaussian_convolution_width", type=float,
                  default=3.0,
                  dest="GAUS_CONV_WIDTH",
                  help="how wide the convolution should be")


# whether or not to shift the seed such that the vertex
# corresponds to the charge weighted median time of the event
parser.add_argument('--center_track_seed', default=True, action=argparse.BooleanOptionalAction)

# whether or not to use multiple vertex seeds: ~factor of 6 slower
parser.add_argument('--use_multiple_vertex_seeds', default=True, action=argparse.BooleanOptionalAction)

# whether or not to pre-scan the time axis to best-match the seed vertex.
parser.add_argument('--prescan_time', default=True, action=argparse.BooleanOptionalAction)


args = parser.parse_args()
print(args)
print("")

# Make code available to python
sys.path.insert(0, args.PATH_TO_REPO)

# Specify correct gpu
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.GPU_INDEX}'

# Import JAX and require double precision.
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

# Other tools.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import TriplePandel stuff
from lib.simdata_i3 import I3SimHandler
from lib.geo import center_track_pos_and_time_based_on_data
from lib.gupta_network_eqx_4comp import get_network_eval_v_fn, get_network_eval_v_fn_f32
from lib.experimental_methods import get_vertex_seeds
from fitting.llh_scanner import get_scanner
from fitting.llh_fitter import get_fitter
from dom_track_eval import get_eval_network_doms_and_track
from likelihood_conv_mpe_w_noise_logsumexp_gupta import get_neg_c_triple_gamma_llh

# A custom color scheme
from palettable.cubehelix import Cubehelix
cx = Cubehelix.make(start=0.3, rotation=-0.5, n=16, reverse=False, gamma=1.0,
     	max_light=1.0,max_sat=0.5, min_sat=1.4).get_mpl_colormap()

# Specify the grid.
dzen = 0.07 # rad
dazi = 0.07 # rad
n_eval = 50 # number of grid points per axes

# Assume 4-component gupta by default
n_hidden = 96
gupta = True
n_comp = 4

if args.NETWORK == "custom":
    network_path = '/mnt/scratch/baburish/TPN-training/gupta_mixture_jax/test_no_penalties_tree_start_epoch_35.eqx'

elif args.NETWORK == "gupta_4comp_reg":
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_4comp_w_penalty_1.e-4/new_model_no_penalties_tree_start_epoch_1000.eqx')

elif args.NETWORK == "gupta_4comp":
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_4comp/new_model_no_penalties_tree_start_epoch_800.eqx')

elif args.NETWORK == "gupta_3comp":
    # a 3 component gupta needs a different import
    from lib.gupta_network_eqx import get_network_eval_v_fn
    n_comp = 3
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_w_penalty_1.e-3/new_model_no_penalties_tree_start_epoch_260.eqx')

elif args.NETWORK == "gamma_3comp":
    # a 3 component gamma needs different imports
    from lib.small_network import get_network_eval_v_fn
    from likelihood_conv_mpe_w_noise_logsumexp import get_neg_c_triple_gamma_llh
    n_comp = 3
    gupta = False
    network_path = os.path.join(args.PATH_TO_REPO, 'data/small_network')

else:
    raise NotImplementedError(f"network {args.NETWORK} not implemnted.")

# Network logic.
try:
    ni = "f64"
    print("Running f64 model")
    eval_network_v = get_network_eval_v_fn(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
except:
    ni = "f32"
    print("Running f32 model")
    eval_network_v = get_network_eval_v_fn_f32(bpath=network_path, dtype=dtype, n_hidden=n_hidden)

# eval_network_v = get_network_eval_v_fn(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=gupta, n_comp=n_comp)

# Get an IceCube event.
#bp = '/home/fast_storage/i3/22645/ftr/'
sim_handler = I3SimHandler(
					os.path.join(args.PATH_TO_INPUT, args.META_FILE_NAME),
                    os.path.join(args.PATH_TO_INPUT, args.PULSES_FILE_NAME),
                    os.path.join(args.PATH_TO_REPO, 'data/icecube/detector_geometry.csv')
				)

meta, pulses = sim_handler.get_event_data(args.EVENT_INDEX)
print(f"muon energy: {meta['muon_energy_at_detector']/1.e3:.1f} TeV")

# Get dom locations, first hit times, and total charges (for each dom).
event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)

# Remove early pulses.
sim_handler.replace_early_pulse(event_data, pulses)
print("n_doms", len(event_data))

# Get MCTruth.
true_pos = jnp.array([meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']])
true_time = meta['muon_time']
true_zenith = meta['muon_zenith']
true_azimuth = meta['muon_azimuth']
true_src = jnp.array([true_zenith, true_azimuth])
print("true direction:", true_src)

# =======================
# SRT noise model
# =======================

def get_srt_noise_weights(expected_pes):
    
    eps = 1e-12
    log10_pes = jnp.log10(jnp.maximum(expected_pes, eps))

    physicsWeight = 1.0 - jnp.exp(
        -jnp.power(
            10.0,
            0.774234
            - 1.02385 * jnp.arctan(0.969486 - 0.577865 * log10_pes)
            + 0.193763 * jnp.arctan(16.2363 + 4.76944 * log10_pes),
        )
    )

    random_arg = -1.36638 - 1.10099 * jnp.arctan(1.08653 + 0.850798 * log10_pes)
    random_arg_clipped = jnp.minimum(0.0, random_arg)
    randomWeight = 1.0 - jnp.exp(-jnp.power(10.0, random_arg_clipped))

    afterWeight = 1.0 - jnp.exp(
        -jnp.power(
            10.0,
            -0.186922
            + 0.946511 * log10_pes
            - 1.08804 * jnp.arctan(1.73241 + 0.760878 * log10_pes),
        )
    )

    preLateWeight = 1.0 - jnp.exp(
        -jnp.power(
            10.0,
            0.144828
            + 0.928378 * log10_pes
            - 1.11346 * jnp.arctan(1.3868 + 0.780568 * log10_pes),
        )
    )

    return physicsWeight, randomWeight, afterWeight, preLateWeight

# =======================
# PDF plotting utilities
# =======================

from lib.gupta import (
    c_multi_gupta_mpe_logprob_midpoint2_stable_v,
    c_multi_gupta_spe_prob_large_sigma_fine_v,
)

def _to1d(x):
    x = jnp.asarray(x)
    return x.reshape((-1,)) if x.ndim == 0 else x

def plot_pdf_for_hit(eval_network_doms_and_track_fn,
                           event_data,          # (N_DOMS, 5) = [x,y,z,time,charge]
                           hit_index,           # int
                           track_direction,     # (2,)
                           track_vertex,        # (3,)
                           track_time,          # scalar
                           t_min=0.0, t_max=1000.0, n_t=2000,
                           sigma_signal=3.0, sigma_noise=1000.0,
                           floor_pdf_height=1.0/6000.0,
                           weights=(1.0-1e-3-1e-2, 1e-2, 1e-3),
                           show_components=True):

    dom_pos_all = jnp.asarray(event_data[:, :3])          # (N,3)
    first_hit_all = jnp.asarray(event_data[:, 3])         # (N,)
    charge_all    = jnp.asarray(event_data[:, 4])         # (N,)

    logits_all, av_all, bv_all, geo_time_all = eval_network_doms_and_track_fn(
        dom_pos_all, track_vertex, track_direction
    )
    # shapes:
    # logits_all: (N, n_comp)
    # av_all    : (N, n_comp, ?)
    # bv_all    : (N, n_comp, ?)
    # geo_time_all: (N,)

    i = int(hit_index)
    first_hit_time = first_hit_all[i]
    charge_i = charge_all[i]

    delay_obs = first_hit_time - (geo_time_all[i] + track_time)
    print("delay_obs: "+str(delay_obs)+" for idx "+str(i))

    log_mix_probs_i = jax.nn.log_softmax(logits_all[i])       # (n_comp,)
    mix_probs_i = jnp.exp(log_mix_probs_i)                    # (n_comp,)
    av_i = av_all[i]                                          # (n_comp, ?)
    bv_i = bv_all[i]                                          # (n_comp, ?)

    log_mix_b = log_mix_probs_i[None, ...]                    # (1, n_comp)
    mix_b     = mix_probs_i[None, ...]                        # (1, n_comp)
    av_b      = av_i[None, ...]                               # (1, n_comp, ?)
    bv_b      = bv_i[None, ...]                               # (1, n_comp, ?)
    nphot_b   = jnp.array([charge_i])                         # (1,)
    sigma_sig = jnp.array(sigma_signal)
    sigma_noi = jnp.array(sigma_noise)

    t_grid = jnp.linspace(t_min, t_max, n_t)                  # (T,)

    # physics logpdf(t)
    def phys_logpdf_at_t(t):
        return c_multi_gupta_mpe_logprob_midpoint2_stable_v(
            jnp.array([t]),   # (1,)
            log_mix_b,        # (1, n_comp)
            av_b,             # (1, n_comp, ?)
            bv_b,             # (1, n_comp, ?)
            nphot_b,          # (1,)
            sigma_sig         # scalar
        )[0]  # (1,) -> scalar

    physics_logpdf = jax.vmap(phys_logpdf_at_t, in_axes=(0,))(t_grid)  # (T,)
    physics_pdf = jnp.exp(physics_logpdf)                               # (T,)

    # noise pdf(t)
    def noise_pdf_at_t(t):
        return c_multi_gupta_spe_prob_large_sigma_fine_v(
            jnp.array([t]),  # (1,)
            mix_b,           # (1, n_comp)
            av_b,            # (1, n_comp, ?)
            bv_b,            # (1, n_comp, ?)
            sigma_noi        # scalar
        )[0]  # (1,) -> scalar

    
    noise_pdf = jax.vmap(noise_pdf_at_t, in_axes=(0,))(t_grid)          # (T,)
    floor_pdf = jnp.ones_like(t_grid) * floor_pdf_height                 # (T,)
    
    # calculate expected pes
    # mix_b: shape (1, n_comp)
    mix_probs = mix_b[0]                  # (n_comp,)
    n_vals = jnp.arange(1, mix_probs.shape[0] + 1)  # [1, 2, ..., n_comp]
    
    expected_pes_raw = jnp.sum(n_vals * mix_probs)
    expected_pes = jnp.maximum(expected_pes_raw, 1e-3)  # prevent zero

    # --- SRT weights ---
    physicsWeight, randomWeight, afterWeight, preLateWeight = \
    get_srt_noise_weights(expected_pes)

    W_tot = physicsWeight + randomWeight + afterWeight + preLateWeight

    w_signal = physicsWeight / W_tot
    w_noise  = (randomWeight + afterWeight + preLateWeight) / W_tot

    # mixture
    w_floor = weights[2]
    
    mixture_pdf = w_signal*physics_pdf + w_noise*noise_pdf + w_floor*floor_pdf
    
    phys_logpdf_obs = phys_logpdf_at_t(delay_obs)
    phys_pdf_obs = jnp.exp(phys_logpdf_obs)
    noise_pdf_obs = noise_pdf_at_t(delay_obs)

    total_pdf_obs = (
        w_signal*phys_pdf_obs
        + w_noise*noise_pdf_obs
        + w_floor*floor_pdf_height
    )
    per_hit_neg2logL = -2.0 * jnp.log(total_pdf_obs)

    print("per_hit_neg2logL: "+str(per_hit_neg2logL))
    '''
    mixture_pdf = w_signal*physics_pdf + w_noise*noise_pdf + w_floor*floor_pdf  # (T,)

    phys_logpdf_obs = phys_logpdf_at_t(delay_obs)
    phys_pdf_obs = jnp.exp(phys_logpdf_obs)
    noise_pdf_obs = noise_pdf_at_t(delay_obs)
    total_pdf_obs = w_signal*phys_pdf_obs + w_noise*noise_pdf_obs + w_floor*floor_pdf_height
    per_hit_neg2logL = -2.0 * jnp.log(total_pdf_obs)
    '''
    
    # plot
    t_np  = np.asarray(t_grid)
    mix_np = np.asarray(mixture_pdf)

    plt.figure(figsize=(7.2, 4.0))
    plt.plot(t_np, mix_np, linewidth=2.0, label="mixture pdf")
    if show_components:
        plt.plot(t_np, np.asarray(physics_pdf), linestyle="--", label="signal pdf")
        plt.plot(t_np, np.asarray(noise_pdf), linestyle=":", label="noise pdf")
        plt.plot(t_np, np.asarray(floor_pdf), linestyle="-.", label="floor")

    plt.axvline(float(delay_obs), linestyle="--", linewidth=1.5,
                label=f"observed delay = {float(delay_obs):.1f} ns")

    plt.xlabel("time residual Δt [ns]")
    plt.ylabel("probability density")
    plt.title(f"hit_idx #{i}   per-hit -2logL = {float(per_hit_neg2logL):.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pdf_scan_ev_{args.EVENT_INDEX}_hit{i}.png")
    plt.close()

    return {
        "hit_index": i,
        "delay_obs_ns": float(delay_obs),
        "per_hit_neg2logL": float(per_hit_neg2logL),
        "t_grid": t_np,
        "mixture_pdf": mix_np,
    }

def quick_plot_many_hit(eval_network_doms_and_track_fn,
                               event_data,
                               track_direction, track_vertex, track_time,
                               hit_indices=None,    
                               topk=None,           
                               outdir=".",
                               prefix="pdf_pulse",
                               t_min=-200.0, t_max=800.0, n_t=2000,
                               sigma_signal=3.0, sigma_noise=1000.0,
                               floor_pdf_height=1.0/6000.0,
                               weights=(1.0-1e-3-1e-2, 1e-2, 1e-3),
                               show_components=True,
                               dpi=200):

    if hit_indices is None:
        if topk is None:
            raise ValueError("Either hit_indices or topk must be provided.")
        charges = np.asarray(event_data[:, 4])
        hit_indices = np.argsort(charges)[::-1][:topk]

    os.makedirs(outdir, exist_ok=True)

    results = []
    for hi in hit_indices:
        fname = f"{prefix}_{int(hi)}.png"
        res = plot_pdf_for_hit(
            eval_network_doms_and_track_fn=eval_network_doms_and_track_fn,
            event_data=event_data,
            hit_index=int(hi),
            track_direction=track_direction,
            track_vertex=track_vertex,
            track_time=track_time,
            t_min=t_min, t_max=t_max, n_t=n_t,
            sigma_signal=sigma_signal, sigma_noise=sigma_noise,
            floor_pdf_height=floor_pdf_height,
            weights=weights,
            show_components=show_components,
        )
        results.append(res)

    return {
        "hit_indices": [int(h) for h in hit_indices],
        "results": results,
        "outdir": outdir,
    }


if args.SEED == "spline_mpe":
    # Use SplineMPE as a seed.
    track_pos = jnp.array([meta['spline_mpe_pos_x'], meta['spline_mpe_pos_y'], meta['spline_mpe_pos_z']])
    track_time = meta['spline_mpe_time']
    track_zenith = meta['spline_mpe_zenith']
    track_azimuth = meta['spline_mpe_azimuth']
    track_src = jnp.array([track_zenith, track_azimuth])

elif args.SEED == "truth":
    track_pos = true_pos
    track_time = true_time
    track_zenith = true_zenith
    track_azimuth = true_azimuth
    track_src = true_src

else:
    raise ValueError(f"seed {args.SEED} not available. Use spline_mpe or truth")

print("seed direction:", np.rad2deg(track_src), "deg")
print("original seed vertex:", track_pos, "m")

centered_track_pos, centered_track_time = track_pos, track_time
if args.center_track_seed:
    print("shifting seed vertex.")
    centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data(event_data, track_pos, track_time, track_src)

print("seed vertex:", centered_track_pos, "m")

fitting_event_data = jnp.array(event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy())

# --- LLH/Fit ---

# Setup likelihood.
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sig=args.GAUS_CONV_WIDTH)

# Potential for additional stability via prescanning optimal vertex time

# First determine the best-fit
fit_llh = get_fitter(
                        neg_llh,
                        use_multiple_vertex_seeds=args.use_multiple_vertex_seeds,
                        prescan_time=args.prescan_time
                    )

# JIT! We want it to be fast.
fit_llh_jit = jax.jit(fit_llh)

# Run the fit
solution = fit_llh_jit(track_src, centered_track_pos, centered_track_time, fitting_event_data)
best_logl, best_direction, best_vertex, best_time= solution

print("")
print("solution found.")
print(f"logl: {best_logl:.3f}")
print(f"direction: {np.rad2deg(best_direction)} deg")
print("")


# --- pdf plot ---

print("Plotting pdfs.")

quick_plot_many_hit(
    eval_network_doms_and_track_fn=eval_network_doms_and_track,
    event_data=fitting_event_data,
    track_direction=best_direction,
    track_vertex=best_vertex,
    track_time=best_time,
    topk=6, prefix="pdf_hit",
    hit_indices=range(0, len(event_data)),
)
