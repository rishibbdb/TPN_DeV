import jax
import jax.numpy as jnp

def remove_early_pulses(eval_network_doms_and_track, data, track_pos, track_dir, track_time):
    crit = -60.0
    _, _, _, geo_times = eval_network_doms_and_track(data[:,:3], track_pos, track_dir)
    delay_times = data[:,4] - geo_times - track_time
    idx = delay_times > crit
    filtered_data = data[idx]
    return filtered_data
