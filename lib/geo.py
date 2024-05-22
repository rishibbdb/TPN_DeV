import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd

#__n_ice_phase = 1.3195;
#__n_ice_group = 1.35634;
__n_ice_phase = 1.30799291638281
__n_ice_group = 1.32548384613875
__n_ice = __n_ice_group
__theta_cherenkov = np.arccos(1/__n_ice_phase)
__sin_theta_cherenkov = np.sin(__theta_cherenkov)
__tan_theta_cherenkov = np.tan(__theta_cherenkov)
__c = 0.299792458 # m / ns
__c_ice = __c/__n_ice_group


def center_track_pos_and_time_based_on_data(event_data: pd.DataFrame, track_pos, track_time, track_dir):
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)
    centered_track_time = np.sum(event_data['charge'] * event_data['time']) / np.sum(event_data['charge'])
    centered_track_pos = track_pos + (centered_track_time - track_time) * __c * track_dir_xyz
    return jnp.array(centered_track_pos), jnp.float64(centered_track_time)


@jax.jit
def geo_time(dom_pos, track_pos, track_dir):
    """
    roughly following https://github.com/icecube/icetray/blob/dde656a29dbd8330e5f54f9260550952f0269bc9/phys-services/private/phys-services/I3Calculator.cxx#L19

    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    # vector from vertex to dom
    v_a = dom_pos - track_pos

    # distance muon travels from track vertex to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # a vector parallel track with length ds
    ds_v = ds * track_dir

    # vector closest approach position to dom yields closest approach distance
    v_d = v_a - ds_v
    dc = jnp.linalg.norm(v_d)

    # distance that the photon travels
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    return (ds - dx + dt * __n_ice_group) / __c

geo_time_v = jax.jit(jax.vmap(geo_time, (0, None, None), 0))


@jax.jit
def cherenkov_cylinder_coordinates(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    # vector from vertex to dom
    v_a = dom_pos - track_pos

    # distance muon travels from track vertex to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # a vector parallel track with length ds
    ds_v = ds * track_dir

    # vector closest approach position to dom yields closest approach distance
    v_d = v_a - ds_v
    dc = jnp.linalg.norm(v_d)

    # vector to closest approach position gives z-component
    v_c = track_pos + ds_v
    v_c_z = v_c[2]

    # distance that the photon travel
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    ### missing: add last return value -> rho angle of track
    return (ds - dx + dt * __n_ice_group) / __c, dc, v_c_z


cherenkov_cylinder_coordinates_v = jax.jit(jax.vmap(cherenkov_cylinder_coordinates, (0, None, None), (0, 0, 0)))


@jax.jit
def closest_distance_dom_track(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """

    # vector track vertex -> dom
    v_a = dom_pos - track_pos
    # vector: closest point on track -> dom
    v_d = v_a - jnp.dot(v_a, track_dir) * track_dir
    dist = jnp.linalg.norm(v_d)
    return dist

# Generalize to matrix input for dom_pos with shape (N_DOMs, 3).
# Output will be in form of (N_DOMs, 1)
closest_distance_dom_track_v = jax.jit(jax.vmap(closest_distance_dom_track, (0, None, None), 0))


@jax.jit
def convert_spherical_to_cartesian_direction(x):
    """
    x = (theta, phi)
    """
    track_theta = x[0]
    track_phi = x[1]
    track_dir_x = jnp.sin(track_theta) * jnp.cos(track_phi)
    track_dir_y = jnp.sin(track_theta) * jnp.sin(track_phi)
    track_dir_z = jnp.cos(track_theta)
    direction = jnp.array([track_dir_x, track_dir_y, track_dir_z])
    return direction

# Generalize to matrix input for x with shape (N_DOMs, 2) for theta and phi angles.
# Output will be in form of (N_DOMs, 3) for dir_x, dir_y, dir_z
convert_spherical_to_cartesian_direction_v = jax.jit(jax.vmap(closest_distance_dom_track, 0, 0))


@jax.jit
def get_xyz_from_zenith_azimuth(x):
    track_dir = convert_spherical_to_cartesian_direction(x)
    y = -1 * track_dir
    return y

get_xyz_from_zenith_azimuth_v = jax.jit(jax.vmap(get_xyz_from_zenith_azimuth, 0, 0))


@jax.jit
def light_travel_time_i3calculator(dom_pos, track_pos, track_dir):
    """
    roughly following https://github.com/icecube/icetray/blob/dde656a29dbd8330e5f54f9260550952f0269bc9/phys-services/private/phys-services/I3Calculator.cxx#L19

    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    dc = closest_distance_dom_track(dom_pos, track_pos, track_dir)

    # vector track support point -> dom
    v_a = dom_pos - track_pos

    # distance muon travels from support point to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # distance that the photon travels
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    return (ds - dx + dt * __n_ice_group) / __c

light_travel_time_i3calculator_v = jax.jit(jax.vmap(light_travel_time_i3calculator, (0, None, None), 0))


@jax.jit
def closest_point_on_track(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """

    # vector track support point -> dom
    v_a = dom_pos - track_pos
    # vector: vector to closest point on track
    v_c = track_pos + jnp.dot(v_a, track_dir) * track_dir
    return v_c

closest_point_on_track_v = jax.jit(jax.vmap(closest_point_on_track, (0, None, None), 0))


@jax.jit
def z_component_closest_point_on_track(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """

    # vector track support point -> dom
    v_a = dom_pos - track_pos
    # vector: vector to closest point on track
    v_c = track_pos + jnp.dot(v_a, track_dir) * track_dir
    return v_c[2]

z_component_closest_point_on_track_v = jax.jit(jax.vmap(z_component_closest_point_on_track, (0, None, None), 0))


@jax.jit
def rho_dom_relative_to_track(dom_pos, track_pos, track_dir):
    """
    clean up and verify!

    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    v1 = dom_pos - track_pos
    closestapproach = track_pos + jnp.dot(v1, track_dir)*track_dir
    v2 = dom_pos - closestapproach
    zdir = jnp.cross(track_dir, jnp.cross(jnp.array([0,0,1]), track_dir))
    positivedir = jnp.cross(track_dir, zdir)
    ypart = v2-v2*jnp.dot(zdir, v2)
    zpart = v2-ypart
    z = jnp.dot(zpart, zdir)
    y = jnp.dot(ypart, positivedir)
    return jnp.arctan2(y,z)

rho_dom_relative_to_track_v = jax.jit(jax.vmap(rho_dom_relative_to_track, (0, None, None), 0))


'''
_recip__speedOfLight = 3.3356409519815204
_n__ = 1.32548384613875
_tan__thetaC = (_n__**2.-1.)**0.5

@jax.jit
def light_travel_time(dom_pos, track_pos, track_dir):
    """
    SplineMPE uses the I3Calculator version above. Differences are small.
    Better use light_travel_time_i3calculator() or cherenkov_cylinder_coordinates()

    Computes the direct, unscattered time it takes for a photon to travel from
    the track to the dom.
    See Eq. 4 of the AMANDA track reco paper https://arxiv.org/pdf/astro-ph/0407044
    """
    closest_dist = closest_distance_dom_track(dom_pos, track_pos, track_dir)

    # vector track support point -> dom
    v_a = dom_pos - track_pos
    # distance muon travels from support point to point of closest approach.
    d1 = jnp.dot(v_a, track_dir)
    # distance that muon travels beyond closest approach until photon hits.
    d2 = closest_dist * _tan__thetaC
    return (d1+d2) * _recip__speedOfLight

# Generalize to matrix input for dom_pos with shape (N_DOMs, 3).
# Output will be in form of (N_DOMs, 1)
light_travel_time_v = jax.jit(jax.vmap(light_travel_time, (0, None, None), 0))
'''



