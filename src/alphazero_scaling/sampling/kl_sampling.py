import numpy as np
import random


def dkl(p, q):
    """
    The Kullback-Leibler divergence between p and q.
    p_i*log(p_i) = 0,  if p_i is almost 0.
    Values are clipped above 10.
    """
    eps = 1e-15
    v = p * np.log(p / (q + eps), out=np.zeros_like(p), where=(p > eps))
    divergence = np.sum(v, axis=p.ndim - 1)
    return np.minimum(divergence, 10)


def duplicate(trajectory, duplicates):
    states = []
    for d, state in zip(duplicates, trajectory.states):
        copies = int(d) + int(random.random() < d % 1)
        states.extend([state for _ in range(copies)])
    return states


class Sampler(object):
    """ A class for sampling states from a trajectory based on their KL divergence.
        States are over/undersampled if DKL(pi||p) is large/small, respectively."""

    def __init__(self, a=15, b=0.01, gamma=0.8):
        self.a = a
        self.b = b
        self.gamma = gamma
        self.dkl_vals = []

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
        sampling_ratio = self.a * av_dkl + self.b
        target = (1 - self.b) / av_dkl
        old_a = self.a
        self.a = self.gamma * self.a + (1 - self.gamma) * target
        self.reset()
        return sampling_ratio, old_a, target

    def reset(self):
        self.dkl_vals = []
