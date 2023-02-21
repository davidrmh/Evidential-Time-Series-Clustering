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
    return params, loglike
