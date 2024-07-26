import jax.numpy as jnp
import jax
import numpy as np

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def c_multi_gamma_mpe_prob(x, logits, a, b, n, sigma, nmax=20, nint=41):
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

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0)+1.e-5, x - diff]))
    xvals = jnp.linspace(xmin, xmax, nint)

    n_pdf = gn.prob(0.5*(xvals[:-1]+xvals[1:]))
    sfs_power_n = jnp.power(g_pdf.survival_function(xvals), n)

    return jnp.sum( n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]) )

c_multi_gamma_mpe_prob_v = jax.vmap(c_multi_gamma_mpe_prob, (0, 0, 0, 0, 0, None, None, None), 0)
