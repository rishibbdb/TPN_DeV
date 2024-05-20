import pandas as pd
import numpy as np

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

    def get_per_dom_summary(self, event_index: int) -> pd.DataFrame:
        meta, pulses = self.get_event_data(event_index)
        df_qtot = pulses[['sensor_id', 'charge']].groupby(by=['sensor_id'], as_index=False).sum()
        df_tmin = pulses[['sensor_id', 'time']].groupby(by=['sensor_id'], as_index=False).min()
        df = df_qtot.merge(self.geo.iloc[df_qtot['sensor_id']], on='sensor_id', how='outer')
        df['time'] = df_tmin['time'].values
        return df
