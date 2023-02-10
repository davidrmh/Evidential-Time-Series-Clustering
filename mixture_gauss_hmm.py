"""
Author: David Montalvan 2-Feb-2023
Simulation of a mixture of HMM with a 2D Gaussian Multivariate conditional Density
Inspired by the synthetic data from the paper:
Integrating hidden Markov models and spectral analysis for sensory time series clustering.
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal as MVN


class MixGaussHMM:
    def __init__(self, init_dist: torch.tensor, trans_mat: torch.tensor, mix_w: torch.tensor,
                 mu: torch.tensor, sigma: torch.tensor):
        self.init_dist = init_dist  # Initial distribution
        self.trans_mat = trans_mat  # Shape (n_states, n_states)
        self.n_states = trans_mat.shape[0]
        self.mix_w = mix_w  # Shape (n_states, n_mix_components)
        self.mu = mu  # Shape (n_states, n_mix_components, n_dim)
        self.sigma = sigma  # Shape (n_states, n_mix_components, n_dim, n_dim)
        self.n_mix = mu.shape[1]
        self.n_dim = mu.shape[2]
        self.mix_comp = self.get_mix_comp()

    def get_mix_comp(self) -> list:
        """
        Create a list of list with the distributions used for the Gaussian Mixture Model
        Each inner list contains the multivariate distributions of the GMM corresponding to a state.
        The length of the list is `self.n_states`. The length of each inner list is `self.n_mix`.

        :return: List
        """
        mix_comp = []
        for i in range(self.n_states):
            mix_comp.append([])
            for j in range(self.n_mix):
                mix_comp[-1].append(MVN(loc=self.mu[i, j, :], covariance_matrix=self.sigma[i, j, :, :]))
        return mix_comp

    def sample(self, n_seq: int = 1, t_max: int = 10, fix_len: bool = True) -> list:
        sequences = []
        for n in range(n_seq):
            seq = []
            # Determine the length of the sequence
            seq_len = t_max if fix_len else np.random.randint(1, high=t_max + 1)

            # Pick initial state
            # To numpy to avoid error with of probabilities not summing one
            z = np.random.choice(self.n_states, p=self.init_dist.numpy())

            # Sample from the corresponding component
            comp_idx = np.random.choice(self.n_mix, p=self.mix_w[z, :].numpy())
            seq.append(self.mix_comp[z][comp_idx].sample())

            while len(seq) < seq_len:
                # Pick next state
                z = np.random.choice(self.n_states, p=self.trans_mat[z, :].numpy())
                # Sample from the corresponding component
                comp_idx = np.random.choice(self.n_mix, p=self.mix_w[z, :].numpy())
                seq.append(self.mix_comp[z][comp_idx].sample())
            sequences.append(torch.vstack(seq))

        return sequences
