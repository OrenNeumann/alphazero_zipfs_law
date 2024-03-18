
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import spawn

from src.alphazero_scaling.alphazero_base import AlphaZero, JOIN_WAIT_DELAY, TrajectoryState
from src.alphazero_scaling.sampling.kl_sampling import Sampler
from collections import Counter, defaultdict
import pickle

"""
An AlphaZero training process that adds states to the buffer in an amount proportional to the KL divergence between the
model prior and the MCTS policy. 

If count_states==True, will save counters of the frequencies of states played, 
and frequencies of states sampled. It will also save dicts for the turn number of each state played and sampled. 
Calling the function with b will change the minimal sampling probability. """

class TrajectoryStateWithMoves(TrajectoryState):
  def __init__(self, move_number, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.move_number = move_number

class AlphaZeroKLSampling(AlphaZero):
    def __init__(self, count_states=False, b=0.01, **kwargs):
        super().__init__(**kwargs)
        self.count_states = count_states
        self.b = b

        self.sampler = Sampler(b=self.b)
        self.played_counter = Counter()
        self.sample_counter = Counter()
        self.turns_played = defaultdict(int)
        self.turns_sampled = defaultdict(int)

    @staticmethod
    def _create_trajectory_state(state, action, policy, root):
        return TrajectoryStateWithMoves(state.move_number(), state.observation_tensor(), state.current_player(),
                                state.legal_actions_mask(), action, policy,
                                root.total_reward / root.explore_count)

    def collect_trajectories(self, model):
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in self.trajectory_generator():
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

            ###
            sampled_states = self.sampler.kl_sampling(trajectory, model)
            if self.count_states:
                self.played_counter.update(str(s.observation) for s in trajectory.states)
                self.sample_counter.update(str(s.observation) for s in sampled_states)
                for s in trajectory.states:
                    self.turns_played[str(s.observation)] += s.move_number
                for s in sampled_states:
                    self.turns_sampled[str(s.observation)] += s.move_number

            # Warm-up, ignore the sampled states on the first step:
            if len(self.replay_buffer) < self.learn_rate:
                sampled_states = trajectory.states

            self.replay_buffer.extend(
                model_lib.TrainInput(
                    s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in sampled_states)
            ###

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
    
    def _print_step(self, logger, step, num_states, num_trajectories, seconds):
        """ Update the sampler and print stats."""
        ratio, a, target = self.sampler.update_hyperparameters()
        super()._print_step(logger, step, num_states, num_trajectories, seconds)
        logger.print("Sampling ratio: {:.5f}. Coeff.: {:.5f}, Target coeff.: {:.5f}.".format(
                ratio, a, target))
        if self.count_states:
            logger.print("Unique states played: {}, unique states sampled: {}.".format(
                len(self.played_counter), len(self.sample_counter)))

    def alpha_zero(self):
        """Start all the worker processes for a full alphazero setup."""
        actors = [spawn.Process(self.actor, kwargs={"num": i})
                  for i in range(self.config.actors)]
        evaluators = [spawn.Process(self.evaluator, kwargs={"num": i})
                      for i in range(self.config.evaluators)]

        def broadcast(msg):
            for proc in actors + evaluators:
                proc.queue.put(msg)

        try:
            self.actors = actors
            self.learner(evaluators=evaluators, broadcast_fn=broadcast)
        except (KeyboardInterrupt, EOFError):
            print("Caught a KeyboardInterrupt, stopping early.")
        finally:
            with open(self.config.path + '/played.pkl', 'wb') as f:
                pickle.dump(self.played_counter, f)
            with open(self.config.path + '/sampled.pkl', 'wb') as f:
                pickle.dump(self.sample_counter, f)
            if self.count_states:
                with open(self.config.path + '/turns_played.pkl', 'wb') as f:
                    pickle.dump(self.turns_played, f)
                with open(self.config.path + '/turns_sampled.pkl', 'wb') as f:
                    pickle.dump(self.turns_sampled, f)
            
            broadcast("")
            # for actor processes to join we have to make sure that their q_in is empty,
            # including backed up items
            for proc in actors:
                while proc.exitcode is None:
                    while not proc.queue.empty():
                        proc.queue.get_nowait()
                    proc.join(JOIN_WAIT_DELAY)
            for proc in evaluators:
                proc.join()