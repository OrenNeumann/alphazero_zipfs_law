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
    """
    def trajectory_generator(self):
        while True:
            found = 0
            for actor_process in self.actors:
                try: ###
                    trajectory = actor_process.queue.get_nowait()
                    if random.random() > self.disable_rate:
                        cut = max(self.cutoff, len(trajectory.states) - self.end_tolerance)
                        trajectory.states = trajectory.states[:cut]
                    yield trajectory
                    ###
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms
    """


    def collect_trajectories(self, model):
        """Collects the trajectories from actors into the replay buffer.
            'model' may be used in derived classes."""
        num_trajectories = 0
        num_states = 0
        for trajectory in self.trajectory_generator():
            ###
            check = True
            l = len(trajectory.states)
            if random.random() > self.disable_rate:
                cut = max(self.cutoff, len(trajectory.states) - self.end_tolerance)
                trajectory.states = trajectory.states[:cut]
            else:
                check = False
            if check:
                if l> cut:
                    if len(trajectory.states) != cut:
                        raise ValueError(f"Cut-off failed: {len(trajectory.states)} -> {l}") 
                    ###
            raise ValueError(f"Cut-off failed: {len(trajectory.states)} -> {l}") 
            num_trajectories += 1
            num_states += len(trajectory.states)
            self.game_lengths.add(len(trajectory.states))
            self.game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                self.outcomes.add(0)
            elif p1_outcome < 0:
                self.outcomes.add(1)
            else:
                self.outcomes.add(2)

            self.replay_buffer.extend(
                model_lib.TrainInput(
                    s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in trajectory.states)

            for stage in range(self.stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (self.stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                self.value_accuracies[stage].add(1 if accurate else 0)
                self.value_predictions[stage].add(abs(n.value))

            if num_states >= self.learn_rate:
                break
        return num_trajectories, num_states

