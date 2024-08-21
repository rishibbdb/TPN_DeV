import jax
import jax.numpy as jnp

def remove_early_pulses(eval_network_doms_and_track, data, track_pos, track_dir, track_time):
    crit = -50.0 # ns
    _, _, _, geo_times = eval_network_doms_and_track(data[:,:3], track_pos, track_dir)
    delay_times = data[:, 3] - geo_times - track_time
    idx = delay_times > crit
    filtered_data = data[idx]
    return filtered_data


def remove_early_pulses_and_noise_candidate_doms(eval_network_doms_and_track, data, track_pos, track_dir, track_time):
    crit_time = -50 # ns
    crit_charge = 0.05 # pe
    _, _, _, geo_times, predicted_charge = eval_network_doms_and_track(data[:,:3], track_pos, track_dir)

    # filter early pulses
    delay_times = data[:, 3] - geo_times - track_time
    idx_time = delay_times > crit_time

    # filter predicted low charger doms
    charges = data[:, 4]
    predicted_charge = jnp.squeeze(jnp.sum(charges) / jnp.sum(predicted_charge) * predicted_charge)
    idx_charge = predicted_charge > crit_charge

    idx = jnp.logical_and(idx_time, idx_charge)
    filtered_data = data[idx]
    return filtered_data


def get_clean_pulses_fn(eval_network_doms_and_track_fn, n_pulses=1):

    def clean_pulses(data, mctruth):
        track_src = mctruth[2:4]
        track_time = mctruth[4]
        track_pos = mctruth[5:8]

        crit = -50.0 # ns
        #_, _, _, geo_times, _ = eval_network_doms_and_track_fn(data[:,:3], track_pos, track_src)
        _, _, _, geo_times = eval_network_doms_and_track_fn(data[:,:3], track_pos, track_src)
        delay_times = data[:, 3] - geo_times - track_time
        idx = delay_times > crit
        idx = idx.reshape((data.shape[0], 1))
        data_clean = jnp.where(idx, data, jnp.zeros((1,3+2*n_pulses)))
        return data_clean

    return clean_pulses


def get_clean_pulses_fn_v(eval_network_doms_and_track_fn, n_pulses=1):
    clean_pulses = get_clean_pulses_fn(eval_network_doms_and_track_fn, n_pulses=n_pulses)
    clean_pulses_v = jax.jit(jax.vmap(clean_pulses, 0, 0))
    return clean_pulses_v


