import jax
import jax.numpy as jnp

def remove_early_pulses(eval_network_doms_and_track, data, track_pos, track_dir, track_time):
    crit = -60.0
    _, _, _, geo_times = eval_network_doms_and_track(data[:,:3], track_pos, track_dir)
    delay_times = data[:, 3] - geo_times - track_time
    idx = delay_times > crit
    filtered_data = data[idx]
    return filtered_data


def get_clean_pulses_fn(eval_network_doms_and_track_fn, orig_length):

    def clean_pulses(data, mctruth):
        track_src = mctruth[2:4]
        track_time = mctruth[5]
        track_pos = mctruth[6:9]

        crit = -60.0
        _, _, _, geo_times = eval_network_doms_and_track_fn(data[:,:3], track_pos, track_src)
        delay_times = data[:, 3] - geo_times - track_time
        idx = delay_times > crit

		# reshape from (Ndoms,)
		# to (Ndoms, 5) aka (Ndoms, Nvariables) via broadcasting
        idx = idx.reshape((orig_length, 1))

		# pad with zeros
        data_clean = jnp.where(idx, data, jnp.zeros((1,5)))
        return data_clean

    return clean_pulses
