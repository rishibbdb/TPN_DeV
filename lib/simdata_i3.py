import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

try:
    import tensorflow as tf
except ImportError:
    print("did not find tensorflow(cpu). can not use batched data loader.")

class I3SimHandlerFtr:
    def __init__(self, events_meta_file: str,
                 events_pulses_file: str,
                 geo_file: str) -> None:

        self.events_meta = pd.read_feather(events_meta_file)
        self.events_data = pd.read_feather(events_pulses_file)
        self.geo = pd.read_csv(geo_file)

    def get_event_data(self, event_index: int) -> pd.DataFrame:
        ev_idx = event_index
        event_meta = self.events_meta.iloc[ev_idx]
        event_data = (self.events_data.iloc[int(event_meta.idx_start): int(event_meta.idx_end + 1)]).copy(deep=True)
        return event_meta, event_data

    def get_per_dom_summary_from_sim_data(self, meta: pd.DataFrame, pulses: pd.DataFrame) -> pd.DataFrame:
        df_qtot = pulses[['sensor_id', 'charge']].groupby(by=['sensor_id'], as_index=False).sum()
        df_tmin = pulses[['sensor_id', 'time']].groupby(by=['sensor_id'], as_index=False).min()
        df = df_qtot.merge(self.geo.iloc[df_qtot['sensor_id']], on='sensor_id', how='outer')
        df['time'] = df_tmin['time'].values
        return df

    def get_per_dom_summary_from_index(self, event_index: int) -> pd.DataFrame:
        meta, pulses = self.get_event_data(event_index)
        df_qtot = pulses[['sensor_id', 'charge']].groupby(by=['sensor_id'], as_index=False).sum()
        df_tmin = pulses[['sensor_id', 'time']].groupby(by=['sensor_id'], as_index=False).min()
        df = df_qtot.merge(self.geo.iloc[df_qtot['sensor_id']], on='sensor_id', how='outer')
        df['time'] = df_tmin['time'].values
        return df


class I3SimBatchHandler:
    @tf.autograph.experimental.do_not_convert
    def __init__(self, sim_handler, process_n_events=None, batch_size=256):
        self.sim_handler = sim_handler
        self.n_events = len(sim_handler.events_meta)
        self.batch_size = batch_size
        if process_n_events is not None:
            self.n_events = process_n_events

        pulse_data = []
        meta_data = []
        n_doms_max = 0
        for i in range(self.n_events):
            ev, meta = self._get_event_data(i)
            pulse_data.append(ev)
            meta_data.append(meta)
            if ev.shape[0] > n_doms_max:
                n_doms_max = ev.shape[0]

        n_doms_max += 1
        pulse_data_tf = tf.ragged.constant(pulse_data, ragged_rank=1, dtype=tf.float64)
        meta_data_tf = tf.constant(meta_data, dtype=tf.float64)

        # TF's batch by sequence length magic
        n_bins = 12
        ds = tf.data.Dataset.from_tensor_slices((pulse_data_tf, meta_data_tf))
        ds = ds.map(lambda x, y: (x, y))
        _element_length_funct = lambda x, y: tf.shape(x)[0]
        ds = ds.bucket_by_sequence_length(
                    element_length_func = _element_length_funct,
                    bucket_boundaries = np.logspace(1, np.log10(n_doms_max), n_bins+1).astype(int).tolist(),
                    bucket_batch_sizes = [self.batch_size]*(n_bins+2),
                    drop_remainder = False,
                    pad_to_bucket_boundary=True
                )

        self.tf_dataset = ds

    def get_batch_iterator(self):
        return iter(self.tf_dataset)


    def _get_event_data(self, event_index):
        meta, pulses = self.sim_handler.get_event_data(event_index)

        # Get dom locations, first hit times, and total charges (for each dom).
        event_data = self.sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)

        return (event_data[['x', 'y','z','time', 'charge']].to_numpy(),
                meta[['muon_energy_at_detector', 'q_tot', 'muon_zenith', 'muon_azimuth', 'muon_time',
                      'muon_pos_x', 'muon_pos_y', 'muon_pos_z', 'spline_mpe_zenith',
                      'spline_mpe_azimuth', 'spline_mpe_time', 'spline_mpe_pos_x',
                      'spline_mpe_pos_y', 'spline_mpe_pos_z']].to_numpy())







