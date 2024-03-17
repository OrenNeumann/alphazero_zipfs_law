
import imp
import itertools
import time
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import spawn

import src.alphazero_scaling.alphazero_base as base
from collections import Counter, defaultdict
import pickle
import numpy as np


"""

In AlphaGo Zero, they cut-off games that were a clear loss, with a false-positive rate of 5%.
They kept an estimate of the false positive rate by playing out 10% of the games after the cutoff point.
"""

class Trajectory(base.Trajectory):
    """A sequence of observations, actions and policies, and the outcomes."""

    def __init__(self):
        super().__init__()
        self.resigned = False



class AlphaZeroWithResignation(base.AlphaZero):
    def __init__(self, gamma=0.9, **kwargs):
        super().__init__(**kwargs)
        self.v_resign = -0.8
        self.gamma = gamma
        self.false_positive_rate = 0
        self.target_rate = 0.05
        self.disable_resign_rate = 0.1

    def _update_cutoff(self):
        delta = self.target_rate - self.false_positive_rate
        target = 0
        self.cutoff = self.gamma * self.cutoff + (1 - self.gamma) * target

    def _play_game(self, logger, game_num, game, bots, temperature, temperature_drop):
        """Play one game, return the trajectory."""
        trajectory = Trajectory()
        actions = []
        state = game.new_initial_state()
        random_state = np.random.RandomState()
        logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
        logger.opt_print("Initial state:\n{}".format(state))
        while not state.is_terminal():
            if state.is_chance_node():
                # For chance nodes, rollout according to chance node's probability
                # distribution
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = random_state.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                """
                if model is not None:
                    # estimate v
                    obs = state.observation_tensor()
                    mask = state.legal_actions_mask()
                    value = model.inference([obs], [mask])[0][0][0] 
                    if state.current_player() == 1:
                        value = -value
                    if value <= self.v_resign:
                        resigned = True
                        if np.random.uniform() < self.disable_resign_rate:
                            pass
                        else:
                            break
                """
                root = bots[state.current_player()].mcts_search(state)
                
                value = root.total_reward / root.explore_count
                if (not trajectory.resigned) and (value <= self.v_resign):
                        trajectory.resigned = True
                        if np.random.uniform() < self.disable_resign_rate:
                            pass
                        else:
                            player_one_value = value * (1 - state.current_player() * 2)
                            returns = [player_one_value, -player_one_value]
                            break
                
                policy = np.zeros(game.num_distinct_actions())
                for c in root.children:
                    policy[c.action] = c.explore_count
                policy = policy ** (1 / temperature)
                policy /= policy.sum()
                if len(actions) >= temperature_drop:
                    action = root.best_child().action
                else:
                    action = np.random.choice(len(policy), p=policy)
                trajectory.states.append(self._create_trajectory_state(state, action, policy, root))
                action_str = state.action_to_string(state.current_player(), action)
                actions.append(action_str)
                logger.opt_print("Player {} sampled action: {}".format(
                    state.current_player(), action_str))
                state.apply_action(action)
        logger.opt_print("Next state:\n{}".format(state))
        if state.is_terminal():
            trajectory.returns = state.returns()
        else:
            trajectory.returns = returns
        logger.print("Game {}: Returns: {}; Actions: {}".format(
            game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
        return trajectory