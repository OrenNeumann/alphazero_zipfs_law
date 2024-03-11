"""
from src.alphazero_scaling.loading import load_model_from_checkpoint, load_config
path = '/mnt/ceph/neumann/alphazero/scratch_backup/models/connect_four_10000/q_0_0/'
config = load_config(path)
model = load_model_from_checkpoint(config=config, path=path, checkpoint_number=10_000)
#game = pyspiel.load_game('connect_four')
"""

"""
changes lines 384-385 in alpha_zero, and feeds the buffer a modified batch of states with
different frequencies:
sampler_trajectory = sampler.kl_sampling(trajectory, model)
replay_buffer.extend(model_lib.TrainInput(s.observation, s.legals_mask, s.policy, p1_outcome) for s in sampler_trajectory.states)

Change line 396 to update a,b
plot in line 440
"""

import numpy as np
import random


def dkl(p, q):
    """
    The Kullback-Leibler divergence between p and q.
    p_i*log(p_i) is 0 if p_i is close to 0.
    """
    v = p * np.log(p / q, out=np.zeros_like(p), where=(p > 1e-15))
    return np.sum(v, axis=v.ndim - 1)


def duplicate(trajectory, duplicates):
    states = []
    for d, state in zip(duplicates, trajectory.states):
        copies = int(d) + int(random.random() < d % 1)
        states.extend([state for _ in range(copies)])
    return states


class Sampler(object):
    """ A class for sampling states from a trajectory based on their KL divergence.
        States are over/undersampled if DKL(pi||p) is large/small, respectively."""

    def __init__(self, a=2, b=0.01, gamma=0.8):
        self.a = a
        self.b = b
        self.gamma = gamma
        self.dkl_vals = []
        self.n_states = 0

    def kl_sampling(self, trajectory, model):
        """
        Sample states from a trajectory, duplicating each state by a factor of:
        a * DKL(pi||p) + b
        Fractions are handled as probabilities.
        """
        policies = []
        observations = []
        masks = []
        for state in trajectory.states:
            observations.append(state.observation)
            masks.append(state.legals_mask)
            policies.append(state.policy)
        priors = model.inference(observations, masks)[1]
        policies = np.array(policies)

        kl_divergence = dkl(policies, priors)

        n_duplicates = self.a * kl_divergence + self.b
        self.dkl_vals.extend(kl_divergence)

        return duplicate(trajectory, n_duplicates)

    def update_hyperparameters(self):
        """ Anneal 'a' to the value that keeps the sampling ratio = 1 (neither over- or undersample)."""
        av_dkl = np.mean(self.dkl_vals)
        sampling_ratio = (self.a * av_dkl + self.b) / self.n_states
        target = (1 - self.b) / av_dkl
        old_a = self.a
        self.a = self.gamma * self.a + (1 - self.gamma) * target
        self.reset()
        return sampling_ratio, old_a, target

    def reset(self):
        self.dkl_vals = []
        self.n_states = 0
