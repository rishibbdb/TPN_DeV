import jax
import jax.numpy as jnp

from lib.geo import cherenkov_cylinder_coordinates_v
from lib.geo import rho_dom_relative_to_track_v
from lib.geo import get_xyz_from_zenith_azimuth
from lib.trafos import transform_network_outputs_v, transform_network_inputs_v


def get_eval_network_doms_and_track(eval_network_v_fn):
    """
    network eval function (vectorized across doms)
    """

    @jax.jit
    def eval_network_doms_and_track(dom_pos, track_vertex, track_dir):
        """
        track_direction: (zenith, azimuth) in radians
        track_vertex: (x, y, z)
        dom_pos: 2D array (n_doms X 3) where columns are x,y,z of dom location
        """
        track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)

        geo_time, closest_approach_dist, closest_approach_z = \
        cherenkov_cylinder_coordinates_v(dom_pos,
                                         track_vertex,
                                         track_dir_xyz)

        closest_approach_rho = rho_dom_relative_to_track_v(dom_pos,
                                                           track_vertex,
                                                           track_dir_xyz)

        track_zenith = track_dir[0]
        track_azimuth = track_dir[1]
        x = jnp.column_stack([closest_approach_dist,
                          closest_approach_rho,
                          closest_approach_z,
                          jnp.repeat(track_zenith, len(closest_approach_dist)),
                          jnp.repeat(track_azimuth, len(closest_approach_dist))])

        x_prime = transform_network_inputs_v(x)
        y_pred = eval_network_v_fn(x_prime)
        logits, av, bv = transform_network_outputs_v(y_pred)
        return logits, av, bv, geo_time

    return eval_network_doms_and_track
