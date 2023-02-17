from dynamax.hidden_markov_model import GaussianHMM
from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import SphericalGaussianHMM
from dynamax.hidden_markov_model import SharedCovarianceGaussianHMM
import hmm_datasets as hd
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Inputs for gathering the data
path = './data/stocks'
old_first = True
sep = '_'
stocks = hd.StockData(path, old_first, sep)
period_len = 15
batches_hlc = hd.batches_norm_hlc_by_open(stocks, period_len)

# Inputs for model specification
num_states = 3 # Downtrending market, uptrending market, sideways market
emi_dim = batches_hlc.shape[-1]

# Fit a GaussianHMM
key = jr.PRNGKey(7)
method = 'prior'

## Create model
gauss_hmm = SharedCovarianceGaussianHMM(num_states = num_states,
                       emission_dim = emi_dim)
## Initialize
params, prop = gauss_hmm.initialize(key = key,
                                   method = method)

## Fit using EM
params, loglike = gauss_hmm.fit_em(params, prop, batches_hlc, num_iters = 10, verbose = True)
plt.plot(loglike, '.-', markersize = 10)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.title('Fitting Gaussian HMM')
plt.grid()
plt.show()
plt.close()
