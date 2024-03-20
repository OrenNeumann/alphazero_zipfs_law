
from open_spiel.python.algorithms.alpha_zero import model as model_lib
import src.alphazero_scaling.alphazero_base as base
from collections import deque
import numpy as np


"""

In AlphaGo Zero, they cut-off games that were a clear loss, with a false-positive rate of 5%.
They kept an estimate of the false positive rate by playing out 10% of the games after the cutoff point.

actors have an object for keeping track of v_resign. updated in update_checkpoint.
learner takes info from trajectories to update v_resign.
"""


class Trajectory(base.Trajectory):
    def __init__(self):
        super().__init__()
        self.test_run = False
        self.resigning_player = None
        self.min_value = 1


class AlphaZeroWithResignation(base.AlphaZero):
    def __init__(self, gamma=0.9, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.n_tests = 0
        self.target_rate = 0.05
        self.disable_resign_rate = 0.1
        self.test_values = deque(maxlen=1000)
        self.test_mask = deque(maxlen=1000)

        self.v_resign = -0.8
        self.target_v = self.v_resign
        self.v_resign_path = self.config.path + 'v_resign.npy'
        with open(self.v_resign_path, 'wb') as f:
            np.save(f, self.v_resign)

    def _update_v_resign(self):
        """ Calculate target for v_resign, then anneal.
            The self.v_resign variable is available only to the learner, and shared 
            with the actors through the v_resign_path file.
        """
        fp_values = np.array(self.test_values)[self.test_mask]
        #fp_values.sort()
        #if len(fp_values) == 0:
        #    return
        """
        target_fp_num = self.target_rate * len(self.test_values)
        if len(fp_values) < target_fp_num: 
            # total num. of wins smaller than 5% of all (tested) resigned games - v_resign is too low.
            # (even if all wins are resigned, it's below 5%)
            #self.target_v = self.v_resign * 0.97
            self.target_v = self.target_v * 0.97
        else: #wins are more than 5% of all (tested) resigned games - find v below which non-losses are exactly 5%.
            self.target_v = fp_values[int(target_fp_num)] # use percentile instead """
        target_percent = 100 * self.target_rate * (len(self.test_values) / max(len(fp_values),1))
        if target_percent > 100 or len(fp_values) == 0:
            # too few positives (even if all are false, it's below 5% of all tests)
            self.target_v = self.target_v * 0.97
        else: # find v that gives exactly 5% false positive fraction.
            self.target_v = np.percentile(fp_values, q=target_percent)
        self.v_resign = self.gamma * self.v_resign + (1 - self.gamma) * self.target_v
        with open(self.v_resign_path, 'wb') as f:
            np.save(f, self.v_resign)
        

    def _play_game(self, logger, game_num, game, bots, temperature, temperature_drop):
        """Play one game, return the trajectory."""
        try:
            with open(self.v_resign_path, 'rb') as f:
                v_resign = np.load(f)
        except:
            v_resign = -1
            logger.print("Failed to load v_resign from file.")
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
                root = bots[state.current_player()].mcts_search(state)
                ###
                # Check resignation threshold, then roll for making a test run:
                value = root.total_reward / root.explore_count #seems like it's current player's value
                if (not trajectory.test_run) and (value <= v_resign):
                    if np.random.uniform() < self.disable_resign_rate:
                        trajectory.test_run = True
                        trajectory.resigning_player = state.current_player()
                    else:
                        player_one_value = value * (1 - state.current_player() * 2)
                        returns = [player_one_value, -player_one_value]
                        break
                if trajectory.test_run and state.current_player() == trajectory.resigning_player:
                    trajectory.min_value = min(trajectory.min_value, value)
                ###
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
        ###
        if state.is_terminal():
            trajectory.returns = state.returns()
        else:
            trajectory.returns = returns
        ###
        logger.print("Game {}: Returns: {}; Actions: {}".format(
            game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
        return trajectory
    
    def collect_trajectories(self, model):
        """Collects the trajectories from actors into the replay buffer.
            'model' may be used in derived classes."""
        num_trajectories = 0
        num_states = 0
        num_tests = 0
        for trajectory in self.trajectory_generator():
            num_trajectories += 1
            num_states += len(trajectory.states)
            self.game_lengths.add(len(trajectory.states))
            self.game_lengths_hist.add(len(trajectory.states))

            ###
            if trajectory.test_run:
                num_tests += 1 # maybe for logger
                self.test_values.append(trajectory.min_value)
                is_false_positive = trajectory.returns[trajectory.resigning_player] >= 0
                self.test_mask.append(is_false_positive)
                #if trajectory.returns[trajectory.resigning_player] >= 0: # False positive
                    #fp_values.append(trajectory.min_value)
            ###

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
        self._update_v_resign()
        self.n_tests = num_tests
        return num_trajectories, num_states

    def _print_step(self, logger, *args, **kwargs):
        super()._print_step(logger, *args, **kwargs)
        logger.print("v_resign: {:.2f}. Target: {:.2f}. New tests: {}. False-positive fraction: {:.1f}%.".format(
            self.v_resign, self.target_v, self.n_tests, 100 * sum(self.test_mask) / max(len(self.test_mask),1)))
