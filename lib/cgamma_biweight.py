import jax.numpy as jnp
import jax
import numpy as np

from jax.scipy.special import gamma, gammaincc

def c_multi_gamma_biweight_prob(x, mix_probs, a, b, sigma=3.0):
    # todo: consider exploring logsumexp trick (potentially more stable)
    # e.g. https://github.com/tensorflow/probability/blob/65f265c62bb1e2d15ef3e25104afb245a6d52429/tensorflow_probability/python/distributions/mixture_same_family.py#L348
    # for now: implement naive mixture probs
    return jnp.sum(mix_probs * c_gamma_biweight_prob(x, a, b, sigma), axis=-1)

c_multi_gamma_prob_v = jax.vmap(c_multi_gamma_biweight_prob, (0, 0, 0, 0, None), 0)


def c_gamma_biweight_prob(x, a, b, sigma=3.0):
    s = 2.5 * sigma
    g_a = gamma(a)
    g_1pa = gamma(1+a)
    g_2pa = gamma(2+a)
    g_4pa = gamma(4+a)

    gincc_a = gammaincc(a, b*(s+x)) * g_a
    gincc_1pa = gammaincc(1+a, b*(s+x))*gamma(1+a)
    gincc_2pa = gammaincc(2+a, b*(s+x))*gamma(2+a)
    gincc_3pa = gammaincc(3+a, b*(s+x))*gamma(3+a)
    gincc_4pa = gammaincc(4+a, b*(s+x))*gamma(4+a)

    gincc_a_m = gammaincc(a, b*(x-s)) * g_a
    gincc_1pa_m = gammaincc(1+a, b*(x-s))*gamma(1+a)
    gincc_2pa_m = gammaincc(2+a, b*(x-s))*gamma(2+a)
    gincc_3pa_m = gammaincc(3+a, b*(x-s))*gamma(3+a)
    gincc_4pa_m = gammaincc(4+a, b*(x-s))*gamma(4+a)

    # branch 0 (-s < t < +s)
    tsum0 = (
                (g_a - gincc_a) * b**4 * (s**4 - 2*s**2*x**2 + x**4)
                + (g_1pa - gincc_1pa) * b**3 * (4*s**2*x - 4*x**3)
                + (g_2pa - gincc_2pa) * (b**2*(6*x**2 - 2*s**2))
                - g_2pa * (8*b*x + 4*a*b*x)
                + g_4pa - gincc_4pa
                + gincc_3pa * 4*b*x
    )

    # branch 1 ( t >= +s)
    tsum1 = (
                (gincc_a_m - gincc_a) * (b**4*s**4 - 2*b**4*s**2*x**2 + b**4*x**4)
                + (gincc_1pa_m - gincc_1pa) * (4*b**3*s**2*x - 4*b**3*x**3)
                + (gincc_2pa- gincc_2pa_m) * (2*b**2*s**2 - 6*b**2*x**2)
                + (gincc_3pa - gincc_3pa_m) * (4*b*x)
                + gincc_4pa_m - gincc_4pa

    )

    # combined branches
    tsum = jnp.where(x < s, tsum0, tsum1)
    # set to 0 outside of support
    tsum = jnp.where(x < -s, 0.0, tsum)

    pre_fac = 15.0/(16*b**4*s**5*g_a)
    return pre_fac * tsum

c_gamma_biweight_prob_v = jax.vmap(c_gamma_biweight_prob, (0, 0, 0, None), 0)


# not implemented below
def c_gamma_biweight_sf(x, a, b, sigma=3.0):
    return None


def c_multi_gamma_biweight_sf(x, mix_probs, a, b, sigma=3.0):
    return None

#c_multi_gamma_sf_v = jax.jit(jax.vmap(c_multi_gamma_sf, (0, 0, 0, 0, None), 0))
