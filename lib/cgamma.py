import jax.numpy as jnp
import jax
import numpy as np


@jax.jit
def c_gamma_prob(x, a, b, sigma=3.0, delta=10.0):
    # x < crit_x - delta => region 4
    # x > crit_x + delta => region 5
    # else: exact evaluation => region 1
    # makes a piecewise defined function
    # that ensures stable gradients.
    crit_x = b * sigma**2

    cond_region1 = jnp.logical_and(x < crit_x+delta, x > crit_x-delta)
    x_region1 = jnp.where(cond_region1, x, crit_x)

    cond_region3 = x >= crit_x+delta
    x_region3 = jnp.where(cond_region3, x, crit_x+delta)

    cond_region4 = x <= crit_x-delta
    x_region4 = jnp.where(cond_region4, x, crit_x-delta)


    yvals_region1 = _c_gamma_region1(x_region1, a, b, sigma=sigma)
    yvals_region3 = _c_gamma_region3(x_region3, a, b, sigma=sigma)
    yvals_region4 = _c_gamma_region4(x_region4, a, b, sigma=sigma)

    result1 = jnp.where(cond_region1, yvals_region1, 0.0)
    result3 = jnp.where(cond_region3, yvals_region3, 0.0)
    result4 = jnp.where(cond_region4, yvals_region4, 0.0)

    return result1 + result3 + result4


@jax.jit
def _c_gamma_region1(x, a, b, sigma=3):
    """
    Implements convolution of gamma distribution with a normal distribution.
    Such distribution arises from adding gaussian noise to samples from a gamma distribution.
    See eq. 7 of arXiv:0704.1706 [astro-ph]. Used for region 1.
    This is the most "exact" calculation, relying on direct eval of hyp1f1
    """

    eta = b*sigma - x/sigma
    s_eta_sq = 0.5 * eta**2 # argument to hyp1f1 is always positive.

    fac1 = (b**a * sigma **(a-1) * jnp.exp(-0.5*(x/sigma)**2)) / 2**(0.5*(1+a))
    s1 = jax.scipy.special.hyp1f1(0.5*a, 0.5, s_eta_sq) / jax.scipy.special.gamma(0.5*(a+1))
    s2 = jax.scipy.special.hyp1f1(0.5*(a+1), 3./2., s_eta_sq) / jax.scipy.special.gamma(0.5*a)
    return fac1 * (s1 - np.sqrt(2)*eta*s2)


@jax.jit
def _c_gamma_region3(x, a, b, sigma=3):
    """
    arXiv:0704.1706, eq. 12
    t >= rho sigma^2, a >= 1
    https://github.com/icecube/icetray/blob/7195b9ad8a76b22e0d7a1e9238147952b1645254/rpdf/private/rpdf/pandel.cxx#L207
    Note: a := ksi
    """

    M_LN2 = jnp.log(2.0)
    rhosigma = b*sigma
    eta = rhosigma - x/sigma
    ksi21 = 2.*a - 1
    ksi212 = ksi21*ksi21
    ksi213 = ksi212*ksi21
    z = jnp.fabs(eta)/jnp.sqrt(2.*ksi21)
    sqrt1plusz2 = jnp.sqrt(1 + z*z)
    k = 0.5*(z*sqrt1plusz2 + jnp.log(z+sqrt1plusz2))
    beta=0.5*(z/sqrt1plusz2 - 1.)
    beta2 = beta*beta
    beta3 = beta2*beta
    beta4 = beta3*beta
    beta5 = beta4*beta
    beta6 = beta5*beta
    n1 = (20.*beta3 + 30.*beta2 + 9.*beta)/12.
    n2 = (6160.*beta6 + 18480.*beta5 + 19404.*beta4 + 8028.*beta3 + 945.*beta2)/288.
    n3 = (27227200.*beta6 + 122522400.*beta5 + 220540320.*beta4 + 200166120.*beta3 +\
          94064328.*beta2 + 20546550.*beta + 1403325.)*beta3/51840.

    sigma2 = sigma*sigma
    delay2 = x*x
    eta2 = eta*eta
    alpha=(-0.5*delay2/sigma2 + 0.25*eta2 - 0.5*a + 0.25 + k*ksi21 - 0.5*jnp.log(sqrt1plusz2) -\
           0.5*a*M_LN2 + 0.5*(a-1.)*jnp.log(ksi21) + a*jnp.log(rhosigma))
    phi = 1. - n1/ksi21 + n2/ksi212 - n3/ksi213

    return jnp.exp(alpha)*phi/jax.scipy.special.gamma(a)/sigma


@jax.jit
def _c_gamma_region4(x, a, b, sigma=3):
    """
    arXiv:0704.1706, eq. 13
    https://github.com/icecube/icetray/blob/e773449cfbb9e505dbcdeb3ae84242505fb7f253/rpdf/private/rpdf/pandel.cxx#L237
    t <= rho sigma^2, a >= 1
    Note: a := ksi
    """

    M_SQRTPI = jnp.sqrt(np.pi)
    M_E = jnp.exp(1.0)
    M_SQRT2 = jnp.sqrt(2.0)
    rhosigma = b*sigma
    eta = rhosigma - x/sigma
    ksi21 = 2.*a - 1
    ksi212 = ksi21*ksi21
    ksi213 = ksi212*ksi21
    z = jnp.fabs(eta)/jnp.sqrt(2.*ksi21)
    sqrt1plusz2 = jnp.sqrt(1 + z*z)
    k = 0.5*(z*sqrt1plusz2 + jnp.log(z+sqrt1plusz2))
    beta=0.5*(z/sqrt1plusz2 - 1.)
    beta2 = beta*beta
    beta3 = beta2*beta
    beta4 = beta3*beta
    beta5 = beta4*beta
    beta6 = beta5*beta
    n1 = (20.*beta3 + 30.*beta2 + 9.*beta)/12.
    n2 = (6160.*beta6 + 18480.*beta5 + 19404.*beta4 + 8028.*beta3 + 945.*beta2)/288.
    n3 = (27227200.*beta6 + 122522400.*beta5 + 220540320.*beta4 + 200166120.*beta3 +\
          94064328.*beta2 + 20546550.*beta + 1403325.)*beta3/51840.

    sigma2 = sigma*sigma
    delay2 = x*x
    eta2 = eta*eta

    u = jnp.power(2.*M_E/ksi21, a/2.)*jnp.exp(-0.25)/M_SQRT2
    psi = 1. + n1/ksi21 + n2/ksi212 + n3/ksi213
    cpandel= jnp.power(rhosigma, a)/sigma * jnp.exp(-0.5*delay2/sigma2+0.25*eta2) / (M_SQRT2*M_SQRTPI)

    return  cpandel * u * jnp.exp(-k*ksi21) * psi / jnp.sqrt(sqrt1plusz2)
