#!/usr/bin/env python

from icecube import dataio, dataclasses
import pandas as pd
import numpy as np

from lib.geo import get_xyz_from_zenith_azimuth
from lib.geo import __theta_cherenkov
__theta_cherenkov_deg = np.rad2deg(__theta_cherenkov)


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


def asym_gaussian(diff, sigma, r):
    t1 = np.exp(-0.5 * (diff / sigma)**2)
    t2 = np.exp(-0.5 * (diff / (r * sigma))**2)
    return np.where(diff > 0, t1, t2)


def get_corrected_charge(dom_pos, q_exp, loss_pos, track_dir):
    # dom_pos: location of single dom. shape = (3,)
    # loss_pos: location of all losses. shape = (N_losses, 3)
    # track_dir: direction of track in cartesian coords. shape = (3,)
    # q_exp: vector of expected charge from each loss for this dom. shape = (N_losses,)
    # returns correction factor for qtot for this dom

    # compute vectors between dom and losses
    # shape: N_loss, 3
    dx = dom_pos - loss_pos
    dx_normed = dx / np.expand_dims(np.linalg.norm(dx, axis=1), axis=1)

    # angle between line from loss to dom and track direction
    delta = np.rad2deg(np.arccos(np.clip(np.dot(dx_normed, track_dir), -1.0, 1.0)))
    diff_angle = __theta_cherenkov_deg - delta

    sigma = 8
    r = 2
    weights = asym_gaussian(diff_angle, sigma, r)
    return np.sum(weights * q_exp) / np.sum(q_exp)


def get_pulse_info_w_qtot_correction(frame,
                                     geo_frame,
                                     event_id,
                                     pulses_key = 'TWSRTHVInIcePulsesIC',
                                     millipede_qexp_key = 'MCMostEnergeticTrack_I3MCTree_ExQ',
                                     track_key = 'MCMostEnergeticTrack'):
    """
    Generates a dictionary containing all pulses for this event.
    And scales down charge for DOMs that are dominated by off-time stochastic losses.
    """

    geo = geo_frame['I3Geometry'].omgeo
    pmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses_key)
    expq_dict = frame[millipede_qexp_key]

    I3MCTree = frame['I3MCTree']
    loss_pos = []
    for p in I3MCTree.get_daughters(frame['MCPrimary1']):
         for loss in I3MCTree.get_daughters(p):
               if not 'Mu' in str(loss.type):
                    pos = loss.pos
                    loss_pos.append([pos.x, pos.y, pos.z])

    track = frame[track_key]
    track_dir_xyz = get_xyz_from_zenith_azimuth([track.dir.zenith, track.dir.azimuth])

    n_pulses = 0
    n_channel = 0
    q_tot = 0.0

    data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'corrected_charge':[], 'is_HLC':[]}

    hlc_doms = set([])
    for omkey, om_pulses in pmap.items():
            n_channel += 1

            # assign sensor index
            om_idx = omkey.om - 1
            string_idx = omkey.string - 1
            sensor_idx = string_idx * 60 + om_idx

            # compute charge correction factor
            pos = geo[omkey].position
            dom_pos = np.array([pos.x, pos.y, pos.z])
            q_exp = np.array(expq_dict[omkey])
            correction_factor = get_corrected_charge(dom_pos, q_exp, loss_pos, track_dir_xyz)

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

                 corrected_charge = charge * correction_factor
                 data['corrected_charge'].append(corrected_charge)
                 data['sensor_id'].append(sensor_idx)
                 data['is_HLC'].append(is_HLC)

    summary = {'n_pulses': n_pulses, 'n_channel': n_channel, 'n_channel_HLC': len(hlc_doms), 'q_tot': q_tot}
    return data, summary
