import random
import src.alphazero_scaling.alphazero_base as base
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from collections import Counter
from collections import defaultdict

"""
Train a model with uniform state sampling, ignoring repetitions in the replay buffer.
"""

class UniqueBuffer(base.Buffer):
    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self, count):
        """ Sample without repetitions, using the most recent copy of each state but averaging the value."""
        counter = defaultdict(lambda: [None, 0, 0])
        for state in self.data:
            key = str(state.observation)
            # Update the latest state, the sum of value, and the count of duplicates
            counter[key] = [state, counter[key][1] + state.value, counter[key][2] + 1]
        # Calculate average value for each unique state
        states = [model_lib.TrainInput(s.observation, s.legals_mask, s.policy, sum_value / count) 
                       for (s, sum_value, count) in counter.values()]
        return random.sample(states, count)
    
    def count_duplicates(self):
        counts = Counter(str(state.observation) for state in self.data)
        duplicates = sum(val - 1 for val in counts.values() if val > 1)
        repeat_observations = sum(1 for val in counts.values() if val > 1)
        return duplicates, repeat_observations
    
class AlphaZeroUniformSampling(base.AlphaZero):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = UniqueBuffer(self.config.replay_buffer_size)

    def _print_step(self, logger, *args, **kwargs):
        n_duplicates, n_repeating = self.replay_buffer.count_duplicates()
        super()._print_step(logger, *args, **kwargs)
        logger.print("Duplicates: {}. Repeating states: {}.".format(
            n_duplicates, n_repeating))

