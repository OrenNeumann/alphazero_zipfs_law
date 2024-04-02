import time
import random
from open_spiel.python.utils import spawn
import src.alphazero_scaling.alphazero_base as base

class AlphaZeroTurnCutoff(base.AlphaZero):
    """
    An AlphaZero training process that cuts-off games after a certain turn.
    Every so often cutoff is disabled, so the agent can learn to play late-game states.
    """
    def __init__(self, cutoff=50, disable_rate=0.01, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.disable_rate = disable_rate

    def trajectory_generator(self):
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in self.actors:
                try:
                    trajectory = actor_process.queue.get_nowait()
                    if random.random() > self.disable_rate:
                        trajectory.states = trajectory.states[:self.cutoff]
                    yield trajectory
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

