from functools import partial
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap
from jax.nn import one_hot
from dynamax.hidden_markov_model import CategoricalHMM

initial_probs = jnp.array([0.5, 0.5])
transition_matrix = jnp.array([[0.95, 0.05],
                               [0.10, 0.90]])
emission_probs = jnp.array([[1/6] * 6, [1/10]*5 + [5/10] ])

num_states = 2 # two possible states for each hidden z_{t}
num_emissions = 1 # Number of components in each emission
num_classes = 6 # Number of possible values each component can take on

hmm = CategoricalHMM(num_states = num_states,
                    emission_dim = num_emissions,
                    num_classes = num_classes)

# Initialize using keyword args (read CategoricalHMM.initialize)
# Notice the reshape part in emission_probs
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

# Filtering, i.e., computing the posterior probability
# of latent state at time `t` given the observations up to
# and including time `t`
# See https://probml.github.io/dynamax/api.html#dynamax.ssm.SSM.filter
posterior = hmm.filter(params, samp_emi)
# Marginal log-likelihood up to final time `T`
marg_loglik = posterior.marginal_loglik 
filt_probs = posterior.filtered_probs
pred_probs = posterior.predicted_probs
    
plt.plot(filt_probs[:, 0], '-')
plt.ylabel('$p(z_{t} = 0 | y_{1:t})$')
plt.xlabel('Time')
plt.grid()
plt.show()
plt.close()
    
    
