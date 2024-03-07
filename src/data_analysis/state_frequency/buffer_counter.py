from collections import deque
from random import sample
from src.data_analysis.state_frequency.state_counter import StateCounter


class BufferCounter(StateCounter):
    """ A StateCounter that samples states from a replay buffer.
        
        New Args:
            sample_unique_states: bool, whether to sample unique states from the buffer rather
            than sample normally.
    """
    def __init__(self,
                 env: str,
                 save_serial=False,
                 save_turn_num=False,
                 save_value=False,
                 cut_early_games=True,
                 sample_unique_states=False):
        super().__init__(env=env,
                         save_serial=save_serial,
                         save_turn_num=save_turn_num,
                         save_value=save_value,
                         cut_early_games=cut_early_games)
        self.sample_unique_states = sample_unique_states
        self.batch_size = 2 ** 10
        self.buffer_size = 2 ** 16
        self.buffer_reuse = 10
        self.sample_threshold = int(self.buffer_size / self.buffer_reuse)
        self.buffer = deque(maxlen=self.buffer_size)
        self.states_added = 0

    def reset_counters(self):
        super().reset_counters()
        self.buffer = deque(maxlen=self.buffer_size)
        self.states_added = 0

    def _update_frequencies(self, keys):
        """ Store keys in the buffer, performing updates when hitting the sample threshold."""
        for key in keys:
            self.buffer.append(key)
            self.states_added += 1
            if self.states_added == self.sample_threshold:
                self.states_added = 0
                super()._update_frequencies(self._sample_buffer())

    def _sample_buffer(self):
        if self.sample_unique_states:
            return self._sample_unique_states()
        else:
            return self._sample_uniformly()

    def _sample_unique_states(self):
        """ Sample uniformly from the set of unique keys."""
        unique_keys = list(set(self.buffer))
        if len(unique_keys) < self.batch_size:
            raise Exception(f'Batch size ({self.batch_size}) is larger than number of unique keys ({len(unique_keys)}).')
        return sample(unique_keys, self.batch_size)

    def _sample_uniformly(self):
        """ Conventional sampling."""
        return sample(self.buffer, self.batch_size)
