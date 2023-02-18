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
norm_stocks = hd.norm_hlc_by_open(stocks, inplace = False)
period_len = 15
batches_hlc = hd.batches_norm_hlc_by_open(stocks, period_len)

# Inputs for model specification
num_states = 3 # Downtrending market, uptrending market, sideways market
emi_dim = batches_hlc.shape[-1]

### Fit a GaussianHMM for the whole market ###
key = jr.PRNGKey(7)
method = 'prior'

## Create model
gauss_hmm = GaussianHMM(num_states = num_states,
                       emission_dim = emi_dim)
## Initialize
params, prop = gauss_hmm.initialize(key = key,
                                   method = method)

## Fit using EM
params, loglike = gauss_hmm.fit_em(params, prop, batches_hlc, num_iters = 60,
                                   verbose = True)
plt.plot(loglike, '.-', markersize = 10)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.title('Fitting Gaussian HMM')
plt.grid()
plt.show()
plt.close()

### Fit a GaussianHMM for each stock series ###

models = {}
symbols = norm_stocks.keys()
num_models = len(norm_stocks)
count = 0
num_states = 3 # Downtrending market, uptrending market, sideways market

for s in symbols:
    count +=  1
    print(f'==== Fitting model for {s} ====\n')
    data = norm_stocks[s].to_numpy()
    #data = hd.create_batches(data, period_len, old_first)
    emi_dim = data.shape[-1]
    
    gauss_hmm = GaussianHMM(num_states = num_states,
                           emission_dim = emi_dim)
    params, prop = gauss_hmm.initialize(key = key,
                                       method = method)
    params, loglike = gauss_hmm.fit_em(params, prop,
                                       data,
                                       num_iters = 50,
                                       verbose = True)
    models[s] = {'params': params,
                 'loglike': loglike,
                 'model': gauss_hmm}
    print(f'\n ==== Model fitted. {num_models - count} remaining to be fitted ====\n')

for s in symbols:
    loglike = models[s]['loglike']
    plt.plot(range(1, len(loglike) + 1), loglike, '.-', markersize = 10)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title(f'Fitting result for {s}')
    plt.grid()
    plt.show()
    plt.close()
    

    
    


    