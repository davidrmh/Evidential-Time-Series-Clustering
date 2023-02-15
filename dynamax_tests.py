from functools import partial
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap
from jax.nn import one_hot
from dynamax.hidden_markov_model import CategoricalHMM
import optax

### INFERENCE USING CATEGORICAL HIDDEN MARKOV MODEL ###
initial_probs = jnp.array([0.5, 0.5])
transition_matrix = jnp.array([[0.95, 0.05],
                               [0.10, 0.90]])

# For each state there is a distribution associated to
# the emission.
# More generally, when initializing the model (see below)
# this parameter must have the shape (num_states, num_emissions, num_clases)
# that is, the entry emission_prob[i, j, k] is the probability of 
# having class k for the j-th emission when being in state i
emission_probs = jnp.array([[1/6] * 6, [1/10]*5 + [5/10] ])

num_states = 2 # two possible states for each hidden z_{t}
num_emissions = 1 # Number of components in each emission
num_classes = 6 # Number of possible values each component can take on

hmm = CategoricalHMM(num_states = num_states,
                    emission_dim = num_emissions,
                    num_classes = num_classes)

# Initialize using keyword args (read CategoricalHMM.initialize)
# Notice the reshape part in emission_probs
# https://probml.github.io/dynamax/api.html#dynamax.hidden_markov_model.CategoricalHMM
params, prop = hmm.initialize(initial_probs = initial_probs,
                             transition_matrix = transition_matrix,
                             emission_probs = emission_probs.reshape(num_states, num_emissions, num_classes))

# Some samples
state = {0: "Honest", 1:"Dishonest"}
num_timesteps = 300
prng = jr.PRNGKey(10)
samp_states, samp_emi = hmm.sample(params, prng, num_timesteps)
colors = {0:'green', 1:'red'}
x_axis = jnp.array(range(1, num_timesteps + 1))
for i in range(num_states):
    idx = samp_states == i
    plt.scatter(x_axis[idx], 
                samp_emi[idx, :],
                c = colors[i],
                label = state[i])
plt.xlabel('Time')
plt.ylabel('Observed value')
plt.legend()
plt.show()
plt.close()

# For more information see the documentation of the class HMMPosteriorFiltered
# https://probml.github.io/dynamax/api.html#dynamax.hidden_markov_model.HMMPosteriorFiltered
posterior = hmm.filter(params, samp_emi)
print(type(posterior))

# Marginal (joint) log-likelihood of the emissions up to (and including) final time `T`
# p(y_1, ..., y_T)
marg_loglik = posterior.marginal_loglik

# Float[Array, 'num_timesteps num_states']. Each row contains p(z_t | y_1, ... , y_t)
filt_probs = posterior.filtered_probs

# Float[Array, 'num_timesteps num_states']. Each row contains p(z_t | y_1, ... y_{t-1})
pred_probs = posterior.predicted_probs
    
plt.plot(filt_probs[:, 0], '-')
plt.ylabel('$p(z_{t} = 0 | y_{1:t})$')
plt.xlabel('Time')
plt.grid()
plt.show()
plt.close()
    
### LEARNING A CATEGORICAL HIDDEN MARKOV MODEL ###
initial_probs = jnp.array([0.5, 0.5])
transition_matrix = jnp.array([[0.95, 0.05],
                               [0.10, 0.90]])

# For each state there is a distribution associated to
# the emission.
# More generally, when initializing the model (see below)
# this parameter must have the shape (num_states, num_emissions, num_clases)
# that is, the entry emission_prob[i, j, k] is the probability of 
# having class k for the j-th emission when being in state i
emission_probs = jnp.array([[1/6] * 6, [1/10]*5 + [5/10] ])

num_states = 2 # two possible states for each hidden z_{t}
num_emissions = 1 # Number of components in each emission
num_classes = 6 # Number of possible values each component can take on

hmm = CategoricalHMM(num_states = num_states,
                    emission_dim = num_emissions,
                    num_classes = num_classes)

# Initialize using keyword args
# Notice the reshape part in emission_probs
# https://probml.github.io/dynamax/api.html#dynamax.hidden_markov_model.CategoricalHMM
params, prop = hmm.initialize(initial_probs = initial_probs,
                             transition_matrix = transition_matrix,
                             emission_probs = emission_probs.reshape(num_states, num_emissions, num_classes))

num_batches = 5
num_timesteps = 5000

# Simulate some samples to train on
# part_sam is a partial function waiting for the key argument of the sample method
# See https://probml.github.io/dynamax/api.html#dynamax.ssm.SSM.sample
part_sam = partial(hmm.sample, params, num_timesteps = num_timesteps)

# Vectorize part_sam
# https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap
samp_fun = vmap(part_sam)

# Finally, we can sample
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html?highlight=jax.random.split
batch_states, batch_emissions = samp_fun(jr.split(jr.PRNGKey(42), num_batches))

print(f'The shape of batch_states is {batch_states.shape}') # (num_batches, num_timesteps)
print(f'The shape of batch_emissions is {batch_emissions.shape}') # (num_batches, num_timesteps, emission_dimension)

# Create a new model and sample some initial parameters
hmm = CategoricalHMM(num_states, num_emissions, num_classes, transition_matrix_stickiness = 10.0)
key =jr.PRNGKey(0)
fbgd_params, fbgd_props = hmm.initialize(key)
print(f'Initial parameters for initial distribution {fbgd_params.initial.probs}')
print(f'Initial parameters for transition matrix {fbgd_params.transitions.transition_matrix}')
print(f'Initial parameters for emission densities {fbgd_params.emissions.probs}')
print(f'---Exploring fbgd_props object (PropertySet)---')
print(f'Parameters of the initial distribution are trainable? => {fbgd_props.initial.probs.trainable}')
print(f'Parameters of the initial distribution constrainer => {fbgd_props.initial.probs.constrainer}')

# Stochastic Gradient Descent
# Tries to minimize the negative log marginal probability of the complete sequence, that is, -log p(y_1, ... y_T)
# https://probml.github.io/dynamax/api.html#dynamax.ssm.SSM.fit_sgd

fbgd_key, key = jr.split(key) # Key is needed for selecting minibatches
sgd_opt = optax.sgd(learning_rate = 1e-2, momentum = 0.95)
batch_size = 2
num_epochs = 400
fbgd_params, fbgd_losses = hmm.fit_sgd(fbgd_params,
                                       fbgd_props,
                                       batch_emissions,
                                      optimizer = sgd_opt,
                                      batch_size = batch_size,
                                      num_epochs = num_epochs,
                                      key = fbgd_key)
print(f"========== Let's analyze fitting results ==========")
plt.plot(fbgd_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
plt.close()