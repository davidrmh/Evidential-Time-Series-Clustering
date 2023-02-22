import numpy as np


def fit_model_em(model,
                 train_data,
                 num_states,
                 init_method,
                 key):
    
    # Emission dimension is deduced from train_data
    emi_dim = train_data.shape[-1]
    
    # Create model
    hmm = model(num_states = num_states,
                emission_dim = emi_dim)
    
    # Initialize
    params, prop = hmm.initialize(key = key,
                                  method = init_method)
    
    # Fit with EM
    params, loglike = hmm.fit_em(params,
                                 prop,
                                 train_data)
    return hmm, params, loglike

def cross_val_states_em(model, states, folds, init_method, key):
    for i, s in enumerate(states):
        print(f'=== Starting fitting with {s} states === \n')
        avgloglike = np.zeros(len(states))
        for f in folds:
            train, test = f
            hmm ,par, loglike = fit_model_em(model, train, s, init_method, key)
            avgloglike[i] += hmm.marginal_log_prob(par, test)
    avgloglike = avgloglike / len(folds)
    return avgloglike