import jax.numpy as jnp
import jax
import numpy as np

from jax.scipy.stats.gamma import sf as gamma_sf
from jax.scipy.stats.norm import pdf as norm_pdf

from lib.gamma_sf_approx import gamma_sf_fast

try:

    from tensorflow_probability.substrates import jax as tfp
    tfd = tfp.distributions

    def c_multi_gamma_mpe_prob(x, logits, a, b, n, sigma):
        g_pdf = tfd.MixtureSameFamily(
                      mixture_distribution=tfd.Categorical(
                          logits=logits
                          ),
                      components_distribution=tfd.Gamma(
                        concentration=a,
                        rate=b,
                        force_probs_to_zero_outside_support=True
                          )
                    )

        gn = tfp.distributions.Normal(
                    x,
                    sigma,
                    validate_args=False,
                    allow_nan_stats=False,
                    name='Normal'
                )

        nmax = 6
        nint = 11
        eps = 1.e-6

        xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
        diff = xmax-x
        xmin = jnp.max(jnp.array([jnp.array(0.0)+eps, x - diff]))
        xvals = jnp.linspace(xmin, xmax, nint)

        n_pdf = gn.prob(0.5*(xvals[:-1]+xvals[1:]))
        sfs_power_n = jnp.power(g_pdf.survival_function(xvals), n)

        return jnp.sum( n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]) )

    c_multi_gamma_mpe_prob_v = jax.vmap(c_multi_gamma_mpe_prob, (0, 0, 0, 0, 0, None), 0)

except ImportError:
    print("could not find tensorflow_probabilty with jax backend.")
    print("will not use c_multi_gamma_mpe_prob.")


def c_multi_gamma_mpe_prob_pure_jax(x, mix_probs, a, b, n, sigma=3.0):
    nmax = 8
    nint = 8
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))

    xvals = jnp.linspace(xmin+eps, xmax, nint)
    n_pdf = norm_pdf(0.5*(xvals[:-1]+xvals[1:]), loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    sfs = jnp.sum(mix_probs_e * gamma_sf(xvals_e, a_e, scale=1./b_e), axis=0)
    sfs_power_n = jnp.power(sfs, n)
    return jnp.sum(n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]))

#def c_multi_gamma_mpe_prob_pure_jax(x, mix_probs, a, b, n, sigma=3.0):
# ### attempt at keeping the peak at 0 within region of integration.
#
#    nmax = 3.5
#    nint = 7
#    eps = 1.e-6
#    split_point = 1.0
#
#    # integrate around gaussian region
#    xmin = jnp.max(jnp.array([split_point, x - nmax * sigma]))
#    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
#
#    xvals = jnp.linspace(xmin, xmax, nint)
#    n_pdf = norm_pdf(0.5*(xvals[:-1]+xvals[1:]), loc=x, scale=sigma)
#
#    a_e = jnp.expand_dims(a, axis=-1)
#    b_e = jnp.expand_dims(b, axis=-1)
#    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)
#
#    xvals_e = jnp.expand_dims(xvals, axis=0)
#    sfs = jnp.sum(mix_probs_e * gamma_sf(xvals_e, a_e, scale=1./b_e), axis=0)
#    sfs_power_n = jnp.power(sfs, n)
#    gaussian_region = jnp.sum(n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]))
#    gaussian_region = jnp.clip(gaussian_region, min=0.0)
#
#    # integrate before gaussian region
#    xvals = jnp.linspace(eps, split_point, nint)
#    xvals_e = jnp.expand_dims(xvals, axis=0)
#    n_pdf = norm_pdf(0.5*(xvals[:-1]+xvals[1:]), loc=x, scale=sigma)
#    sfs = jnp.sum(mix_probs_e * gamma_sf(xvals_e, a_e, scale=1./b_e), axis=0)
#    sfs_power_n = jnp.power(sfs, n)
#    before_region = jnp.sum(n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]))
#    before_region = jnp.clip(before_region, min=0.0)
#
#    return gaussian_region+before_region

c_multi_gamma_mpe_prob_pure_jax_v = jax.vmap(c_multi_gamma_mpe_prob_pure_jax, (0, 0, 0, 0, 0, None), 0)


def c_multi_gamma_mpe_prob_pure_jax_fast(x, mix_probs, a, b, n, sigma=3.0):
    nmax = 20
    nint = 81
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))

    xvals = jnp.linspace(xmin+eps, xmax, nint)
    n_pdf = norm_pdf(0.5*(xvals[:-1]+xvals[1:]), loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    sfs = jnp.sum(mix_probs_e * gamma_sf_fast(xvals_e, a_e, b_e), axis=0)
    sfs_power_n = jnp.power(sfs, n)
    return jnp.sum(n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]))

c_multi_gamma_mpe_prob_pure_jax_fast_v = jax.vmap(c_multi_gamma_mpe_prob_pure_jax_fast, (0, 0, 0, 0, 0, None), 0)
