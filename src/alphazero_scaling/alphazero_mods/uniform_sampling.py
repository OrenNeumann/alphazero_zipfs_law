import random
import src.alphazero_scaling.alphazero_base as base


class UniqueBuffer(base.Buffer):
    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self, count):
        """ Convert data to a dict, keeping only the latest copy of each state."""
        unique_dict = {}
        for state in self.data:
            unique_dict[state.observation] = state
        return random.sample(list(unique_dict.values()), count)
    
class AlphaZeroUniformSampling(base.AlphaZero):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = UniqueBuffer(self.config.replay_buffer_size)
