from collections import deque
from random import sample
from src.data_analysis.state_frequency.state_counter import StateCounter


class BufferCounter(StateCounter):
    """ A StateCounter that samples states from a replay buffer.
    """
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
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
            self._add_to_buffer(key)
            if self.states_added == self.sample_threshold:
                self.states_added = 0
                super()._update_frequencies(keys=self._sample_buffer())

    def _add_to_buffer(self, key):
        self.buffer.append(key)
        self.states_added += 1

    def _sample_buffer(self):
        data = list()
        for _ in range(len(self.buffer) // self.batch_size):
            data.extend(sample(self.buffer, self.batch_size))
        return data


class UniqueBufferCounter(BufferCounter):
    """ A BufferCounter that samples unique states from the buffer, rather
        than sample normally.."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _sample_buffer(self):
        """ Sample uniformly from the set of unique keys."""
        unique_keys = list(set(self.buffer))
        if len(unique_keys) < self.batch_size:
            raise Exception(f'Batch size ({self.batch_size}) is larger than number of unique keys ({len(unique_keys)}).')
        # Data is fed in batches, leftover gets discarded:
        missed = len(unique_keys) % self.batch_size
        return sample(unique_keys, len(unique_keys) - missed)
        # Not up to date:
        #return sample(unique_keys, self.batch_size)
    
"""
class ValueSurpriseCounter(BufferCounter):
    def __init__(self, inference_model=None,**kwargs):
        super().__init__(**kwargs)
        self.inference_model = inference_model

    def collect_data(self, **kwargs):
        if self.inference_model is None:
            raise Exception('Please provide an inference model.')
        super().collect_data(**kwargs)

    def _add_to_buffer(self, key):
        "" Only add states that have high surprise (= model value prediction is far from ground truth value).""
        surprise = 0
        if surprise > 0.5:
            self.buffer.append(key)
        self.states_added += 1   
"""

    
        

