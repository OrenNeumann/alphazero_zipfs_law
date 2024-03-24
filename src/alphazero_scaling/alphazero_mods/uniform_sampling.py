import random
import src.alphazero_scaling.alphazero_base as base
from collections import Counter

class UniqueBuffer(base.Buffer):
    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self, count):
        """ Convert data to a dict, keeping only the latest copy of each state."""
        unique_dict = {str(state.observation): state for state in self.data}
        return random.sample(list(unique_dict.values()), count)
    
    def count_duplicates(self):
        """ Count the number of duplicate states in the buffer."""
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

