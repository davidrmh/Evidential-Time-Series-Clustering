import numpy as np
import jax.numpy as jnp


def fit_model_em(model,
                 train_data,
                 num_states,
                 init_method,
                 key,
                 **kwargs):
    
    # Emission dimension is deduced from train_data
    emi_dim = train_data.shape[-1]
    
    # Create model
    hmm = model(num_states = num_states,
                emission_dim = emi_dim)
    
    # Initialize
    params, prop = hmm.initialize(key = key,
                                  method = init_method, **kwargs)
    
    # Fit with EM
    params, loglike = hmm.fit_em(params,
                                 prop,
                                 train_data, verbose = False)
    return hmm, params, loglike

def cross_val_states_em(model, states, folds, init_method, key, **kwargs):
    avgloglike = np.zeros(len(states))
    for i, s in enumerate(states):
        print(f'\n === Starting fitting with {s} states === \n')
        for f in folds:
            train, test = f
            hmm ,par, loglike = fit_model_em(model, train, s, init_method, key, **kwargs)
            avgloglike[i] += hmm.marginal_log_prob(par, test)
    avgloglike = avgloglike / len(folds)
    return avgloglike

def psi(mu1, mu2, cov1, cov2, beta = 0.5):
    cov1_inv = jnp.linalg.inv(cov1)
    cov2_inv = jnp.linalg.inv(cov2)
    cov_dag = jnp.linalg.inv(cov1_inv + cov2_inv)
    mu_dag = cov1_inv @ mu1 + cov2_inv @ mu2
    det_cov_dag = jnp.linalg.det(cov_dag)
    det_cov1 = jnp.linalg.det(cov1)
    det_cov2 = jnp.linalg.det(cov2)
    
    # To avoid nan
    if det_cov1 == 0.0 or det_cov2 == 0.0:
        return 
    
    quad_cov1 = mu1.T @ cov1_inv @ mu1
    quad_cov2 = mu2.T @ cov2_inv @ mu2
    quad_cov_dag = mu_dag.T @ cov_dag @ mu_dag
    factor1 = (det_cov_dag ** 0.5) * (det_cov1 ** (-0.5 * beta)) * (det_cov2 ** (-0.5 * beta))
    factor2 = jnp.exp(-0.5 * beta * (quad_cov1 + quad_cov2 - quad_cov_dag))
    return factor1 * factor2

def product_kernel(param1, param2, period_len, beta = 0.5):
    # Initial distributions
    init1 = param1.initial.probs
    init2 = param2.initial.probs
    
    
    
