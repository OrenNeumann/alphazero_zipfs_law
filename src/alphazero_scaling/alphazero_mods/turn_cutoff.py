import time
import random
from open_spiel.python.utils import spawn
import src.alphazero_scaling.alphazero_base as base
from open_spiel.python.algorithms.alpha_zero import model as model_lib


class AlphaZeroTurnCutoff(base.AlphaZero):
    """
    An AlphaZero training process that cuts-off games after a certain turn.
    'cutoff' is the number of turns before the end of the game to stop counting states.
    'end_tolerance' shortens games by cutting off the last few turns (note that the 
    terminal state is not included in training).
    Cutoff is disabled every so often, so the agent can learn to play late-game states.
    """
    def __init__(self, cutoff=50, disable_rate=0.01, end_tolerance=0, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.end_tolerance = end_tolerance
        self.disable_rate = disable_rate

    def trajectory_generator(self):
        while True:
            found = 0
            for actor_process in self.actors:
                try: ###
                    trajectory = actor_process.queue.get_nowait()
                    if random.random() > self.disable_rate:
                        cut = min(self.cutoff, len(trajectory.states) - self.end_tolerance)
                        trajectory.states = trajectory.states[:cut]
                    yield trajectory
                    ###
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms
