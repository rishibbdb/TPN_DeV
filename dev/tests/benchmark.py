#!/usr/bin/env python

import numpy as np
np.random.seed(0)

import sys

sys.path.insert(0, '/home/storage2/hans/jax_reco/python')
from network import TriplePandleNet
from trafos import transform_network_outputs, transform_dimensions

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
import jax.numpy as jnp
import jax

from jax import config
config.update("jax_enable_x64", True)

# load network
net = TriplePandleNet('../../data/network/')

dist = 25
z = -210
rho = 0.0
zenith = 90.0
azimuth = 0.0
x = transform_dimensions(dist, rho, z, zenith, azimuth)

# force some evaluations in parallel on the gpu
n_doms = 500
x = np.array(x)
xx = x[np.newaxis, :]
xx = np.repeat(xx, n_doms, axis=0)
y = np.random.normal(0.025, 0.001, 500)
xx[:, 0] = y
xx = jnp.array(xx)
xx.devices() # shape n_doms x 7 (n_inputs)

# evaluate all pdfs at 20ns
t = 20*np.ones(n_doms).reshape(1, n_doms)
t = jnp.array(t)
t.devices() # shape 1 x n_doms

@jax.jit
def do_likelihood_for_some_number_of_doms(x):
    # 500 NN evaluations
    z = net.eval_on_batch(x)
    logits, a, b = transform_network_outputs(z)

    # 500 PDF evaluations
    dist = tfd.Independent(
        distribution = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                        logits=logits
                      ),
                  components_distribution=tfd.Gamma(
                        concentration=a,
                        rate=b,
                        force_probs_to_zero_outside_support=True
                      )
                ),
        reinterpreted_batch_ndims=1
    )
    return -dist.log_prob(t)


for i in range(10000):
    if (i%1000) == 0:
        print(i)
    do_likelihood_for_some_number_of_doms(xx)

print("done!")



