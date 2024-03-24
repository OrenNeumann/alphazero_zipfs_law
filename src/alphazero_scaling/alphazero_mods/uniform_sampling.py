import random
import src.alphazero_scaling.alphazero_base as base
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from collections import Counter
from collections import defaultdict

"""
Train a model with uniform state sampling, ignoring repetitions in the replay buffer.
There are 2 improvements here on the base algorithm:

1. The replay buffer is sampled without repeating states. Copies of the same state are averaged together.

2. Each state in the buffer is sampled only once during optimization. In the base algorithm, data is sampled 
several (~64) times from the buffer every optimization step, so that the total number of samples is the size 
of the buffer. This adds a lot of redundancy in the trained data because of repeating samples.
Here, the buffer is shuffled once and then split into batches, so all states in each batch are unique.
"""

class UniqueBuffer(base.Buffer):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.unique_states = list()

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

    def learn(self, step, logger, model):
        """Sample batches without repetitions"""
        losses = []
        batch_size = self.config.train_batch_size
        # Sample the entire (shuffled) buffer:
        buffer_data = self.replay_buffer.sample(len(self.replay_buffer))
        # Feed batches to the model:
        for i in range(len(buffer_data) // batch_size):
            data = buffer_data[i * batch_size: (i + 1) * batch_size]
            losses.append(model.update(data))

        save_path = model.save_checkpoint(
            step if step % self.config.checkpoint_freq == 0 else -1)
        losses = sum(losses, model_lib.Losses(0, 0, 0)) / len(losses)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)
        return save_path, losses

