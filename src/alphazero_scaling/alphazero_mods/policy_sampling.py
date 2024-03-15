

import collections
import datetime
import functools
import itertools
import json
import os
import random
import sys
import tempfile
import time
import traceback

from src.alphazero_scaling.alphazero_base import AlphaZero, watcher, Buffer
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import spawn
from open_spiel.python.utils import stats

from src.alphazero_scaling.sampling.kl_sampling import Sampler
#from sampling.kl_sampling import Sampler
from collections import Counter, defaultdict, namedtuple
import pickle

"""
An AlphaZero training process that adds states to the buffer in an amount proportional to the KL divergence between the
model prior and the MCTS policy. 

If count_states==True, will save counters of the frequencies of states played, 
and frequencies of states sampled. It will also save dicts for the turn number of each state played and sampled. 
Calling the function with b will change the minimal sampling probability. """

COUNTERS = namedtuple('COUNTERS', ['played', 'sampled', 'turns_played', 'turns_sampled'])


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

    def collect_trajectories(self, game_lengths, game_lengths_hist, outcomes, replay_buffer,
                             value_accuracies, value_predictions):
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in self.trajectory_generator():
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            replay_buffer.extend(
                model_lib.TrainInput(
                    s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in trajectory.states)

            for stage in range(self.stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (self.stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= self.learn_rate:
                break
        return num_trajectories, num_states

    @watcher
    def learner(self, *, evaluators, broadcast_fn, logger):
        """A learner that consumes the replay buffer and trains the network."""
        logger.also_to_stdout = True
        replay_buffer = Buffer(self.config.replay_buffer_size)
        logger.print("Initializing model")
        model = self._init_model_from_config(self.config)
        logger.print("Model type: %s(%s, %s)" % (self.config.nn_model, self.config.nn_width,
                                                 self.config.nn_depth))
        logger.print("Model size:", model.num_trainable_variables, "variables")
        save_path = model.save_checkpoint(0)
        logger.print("Initial checkpoint:", save_path)
        broadcast_fn(save_path)

        data_log = data_logger.DataLoggerJsonLines(self.config.path, "learner", True)

        value_accuracies = [stats.BasicStats() for _ in range(self.stage_count)]
        value_predictions = [stats.BasicStats() for _ in range(self.stage_count)]
        game_lengths = stats.BasicStats()
        game_lengths_hist = stats.HistogramNumbered(self.game.max_game_length() + 1)
        outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
        evals = [Buffer(self.config.evaluation_window) for _ in range(self.config.eval_levels)]
        total_trajectories = 0

        last_time = time.time() - 60
        for step in itertools.count(1):
            for value_accuracy in value_accuracies:
                value_accuracy.reset()
            for value_prediction in value_predictions:
                value_prediction.reset()
            game_lengths.reset()
            game_lengths_hist.reset()
            outcomes.reset()

            num_trajectories, num_states = self.collect_trajectories(game_lengths, game_lengths_hist, outcomes,
                                                                     replay_buffer, value_accuracies, value_predictions)
            total_trajectories += num_trajectories
            now = time.time()
            seconds = now - last_time
            last_time = now

            ratio, a, target = self.sampler.update_hyperparameters() ###

            logger.print("Step:", step)
            logger.print(
                ("Collected {:5} states from {:3} games, {:.1f} states/s. "
                 "{:.1f} states/(s*actor), game length: {:.1f}").format(
                    num_states, num_trajectories, num_states / seconds,
                                                  num_states / (self.config.actors * seconds),
                                                  num_states / num_trajectories))
            logger.print("Buffer size: {}. States seen: {}".format(
                len(replay_buffer), replay_buffer.total_seen))
            ###
            logger.print("Sampling ratio: {:.5f}. Coeff.: {:.5f}, Target coeff.: {:.5f}.".format(
                ratio, a, target))
            if self.count_states:
                logger.print("Unique states played: {}, unique states sampled: {}.".format(
                    len(self.played_counter), len(self.sample_counter)))
            ###

            save_path, losses = self.learn(step, logger, replay_buffer, model)

            for eval_process in evaluators:
                while True:
                    try:
                        difficulty, outcome = eval_process.queue.get_nowait()
                        evals[difficulty].append(outcome)
                    except spawn.Empty:
                        break

            batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
            batch_size_stats.add(1)
            data_log.write({
                "step": step,
                "total_states": replay_buffer.total_seen,
                "states_per_s": num_states / seconds,
                "states_per_s_actor": num_states / (self.config.actors * seconds),
                "total_trajectories": total_trajectories,
                "trajectories_per_s": num_trajectories / seconds,
                "queue_size": 0,  # Only available in C++.
                "game_length": game_lengths.as_dict,
                "game_length_hist": game_lengths_hist.data,
                "outcomes": outcomes.data,
                "value_accuracy": [v.as_dict for v in value_accuracies],
                "value_prediction": [v.as_dict for v in value_predictions],
                "eval": {
                    "count": evals[0].total_seen,
                    "results": [sum(e.data) / len(e) if e else 0 for e in evals],
                },
                "batch_size": batch_size_stats.as_dict,
                "batch_size_hist": [0, 1],
                "loss": {
                    "policy": losses.policy,
                    "value": losses.value,
                    "l2reg": losses.l2,
                    "sum": losses.total,
                },
                "cache": {  # Null stats because it's hard to report between processes.
                    "size": 0,
                    "max_size": 0,
                    "usage": 0,
                    "requests": 0,
                    "requests_per_s": 0,
                    "hits": 0,
                    "misses": 0,
                    "misses_per_s": 0,
                    "hit_rate": 0,
                },
            })
            logger.print()

            if 0 < self.config.max_steps <= step:
                break

            broadcast_fn(save_path)

