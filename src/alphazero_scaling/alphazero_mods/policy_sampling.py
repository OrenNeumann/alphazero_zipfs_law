
import itertools
import time
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import spawn

from src.alphazero_scaling.alphazero_base import AlphaZero, watcher, Buffer, JOIN_WAIT_DELAY, TrajectoryState
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

    @watcher
    def learner(self, *, evaluators, broadcast_fn, logger):
        """A learner that consumes the replay buffer and trains the network."""
        logger.also_to_stdout = True
        # Model must be initialized here, making it a class variable makes the run freeze
        logger.print("Initializing model")
        model = self._init_model_from_config(self.config)
        logger.print("Model type: %s(%s, %s)" % (self.config.nn_model, self.config.nn_width,
                                                 self.config.nn_depth))
        logger.print("Model size:", model.num_trainable_variables, "variables")
        save_path = model.save_checkpoint(0)
        logger.print("Initial checkpoint:", save_path)
        broadcast_fn(save_path)

        data_log = data_logger.DataLoggerJsonLines(self.config.path, "learner", True)

        evals = [Buffer(self.config.evaluation_window) for _ in range(self.config.eval_levels)]
        total_trajectories = 0

        last_time = time.time() - 60
        for step in itertools.count(1):
            for value_accuracy in self.value_accuracies:
                value_accuracy.reset()
            for value_prediction in self.value_predictions:
                value_prediction.reset()
            self.game_lengths.reset()
            self.game_lengths_hist.reset()
            self.outcomes.reset()

            num_trajectories, num_states = self.collect_trajectories(model)
            total_trajectories += num_trajectories
            now = time.time()
            seconds = now - last_time
            last_time = now

            ###
            ratio, a, target = self.sampler.update_hyperparameters()
            ###

            logger.print("Step:", step)
            logger.print(
                ("Collected {:5} states from {:3} games, {:.1f} states/s. "
                 "{:.1f} states/(s*actor), game length: {:.1f}").format(
                    num_states, num_trajectories, num_states / seconds,
                                                  num_states / (self.config.actors * seconds),
                                                  num_states / num_trajectories))
            logger.print("Buffer size: {}. States seen: {}".format(
                len(self.replay_buffer), self.replay_buffer.total_seen))
            ###
            logger.print("Sampling ratio: {:.5f}. Coeff.: {:.5f}, Target coeff.: {:.5f}.".format(
                ratio, a, target))
            if self.count_states:
                logger.print("Unique states played: {}, unique states sampled: {}.".format(
                    len(self.played_counter), len(self.sample_counter)))
            ###

            save_path, losses = self.learn(step, logger, model)

            for eval_process in evaluators:
                while True:
                    try:
                        difficulty, outcome = eval_process.queue.get_nowait()
                        evals[difficulty].append(outcome)
                    except spawn.Empty:
                        break

            self._dump_statistics(logger, data_log, step, num_states, seconds, total_trajectories,
                                  num_trajectories, evals, losses)

            if 0 < self.config.max_steps <= step:
                break

            broadcast_fn(save_path)

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