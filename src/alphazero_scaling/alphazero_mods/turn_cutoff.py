import time
import random
import itertools
import numpy as np
from open_spiel.python.utils import spawn
import src.alphazero_scaling.alphazero_base as base
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib


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


class AlphaZeroCaptureCutoff(base.AlphaZero):
    """
    An AlphaZero training process that cuts-off games when a player gets a major advantage.
    'cutoff' is the number of pieces surplus one player has over the other, when the game is cut-off.
    Cutoff is disabled every so often, depending on rate.
    """
    def __init__(self, cutoff=10, disable_rate=0.01, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.disable_rate = disable_rate

    def capture_diff(self, state_str):
        if self.config.game == 'oware':
            score_x = int(state_str.split('\n')[0][17:19])
            score_o = int(state_str.split('\n')[-2][17:19])
        else:
            raise NameError('Environment '+ self.config.game + ' not supported')
        return np.abs(score_x - score_o)
    
    def _play_game(self, logger, game_num, game, bots, temperature, temperature_drop, capture_cut=False):
        """Play one game, return the trajectory."""
        trajectory = base.Trajectory()
        trajectory_cut = base.Trajectory() ###
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
                ###
                if self.capture_diff(str(state)) < self.cutoff:
                    trajectory_cut.states.append(self._create_trajectory_state(state, action, policy, root))
                ###
                action_str = state.action_to_string(state.current_player(), action)
                actions.append(action_str)
                logger.opt_print("Player {} sampled action: {}".format(
                    state.current_player(), action_str))
                state.apply_action(action)
        logger.opt_print("Next state:\n{}".format(state))

        trajectory.returns = state.returns()
        trajectory_cut.returns = state.returns() ###
        logger.print("Game {}: Returns: {}; Actions: {}".format(
            game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
        ###
        if capture_cut:
            return trajectory, trajectory_cut 
        ###
        return trajectory
    
    @base.watcher
    def actor(self, *, logger, queue):
        """An actor process runner that generates games and returns trajectories."""
        logger.print("Initializing model")
        model = self._init_model_from_config(self.config)
        logger.print("Initializing bots")
        az_evaluator = evaluator_lib.AlphaZeroEvaluator(self.game, model)
        bots = [
            self._init_bot(self.config, self.game, az_evaluator, False),
            self._init_bot(self.config, self.game, az_evaluator, False),
        ]
        for game_num in itertools.count():
            if not self.update_checkpoint(logger, queue, model, az_evaluator):
                return
            queue.put(self._play_game(logger, game_num, self.game, bots, self.config.temperature,
                                      self.config.temperature_drop, capture_cut=True)) ###

    def trajectory_generator(self):
        while True:
            found = 0
            for actor_process in self.actors:
                try: ###
                    trajectory, trajectory_cut = actor_process.queue.get_nowait()
                    if random.random() > self.disable_rate:
                        yield trajectory_cut
                    else:
                        yield trajectory
                    ###
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

