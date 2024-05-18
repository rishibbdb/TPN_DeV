#!/usr/bin/env python

from icecube import dataio, dataclasses
import pandas as pd
import numpy as np


def get_pulse_info(frame, event_id, pulses_key = 'TWSRTHVInIcePulsesIC'):
    """
    Generates a dictionary containing all pulses for this event.
    """
    pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses_key)
    n_pulses = 0
    n_channel = 0
    q_tot = 0.0

    data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'is_HLC':[]}

    hlc_doms = set([])
    for omkey, om_pulses in pulses.items():
            n_channel += 1

            # assign sensor index
            om_idx = omkey.om - 1
            string_idx = omkey.string - 1
            sensor_idx = string_idx * 60 + om_idx

            for i, pulse in enumerate(om_pulses):
                 n_pulses += 1
                 time = pulse.time
                 charge = pulse.charge
                 is_HLC = int(pulse.flags & dataclasses.I3RecoPulse.PulseFlags.LC)
                 if is_HLC:
                    q_tot += charge
                    if not omkey in hlc_doms:
                        hlc_doms.add(omkey)

                 # store pulse data
                 data['event_id'].append(event_id)
                 data['time'].append(time)
                 data['charge'].append(charge)
                 data['sensor_id'].append(sensor_idx)
                 data['is_HLC'].append(is_HLC)

    summary = {'n_pulses': n_pulses, 'n_channel': n_channel, 'n_channel_HLC': len(hlc_doms), 'q_tot': q_tot}
    return data, summary

