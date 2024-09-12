"""
Tools for getting an estimate of a state's value. Either from a trained agent, or a solver.
"""

from src.alphazero_scaling.solver_bot import connect_four_solver
from src.alphazero_scaling.loading import load_model_from_checkpoint, load_config
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms import mcts
import numpy as np
import numpy.typing as npt
import pyspiel
from tqdm import tqdm
import pickle


def get_model_value_estimator(env: str, config_path: str):
    """ Creates an estimator function that uses a trained model to calculate state values.

        :param env: Game environment.
        :param config_path: Path to trained model.
    """
    game = pyspiel.load_game(env)
    config = load_config(config_path)
    model = load_model_from_checkpoint(config=config, path=config_path, checkpoint_number=10_000)

    def model_value(serial_states: list[str]) -> npt.NDArray[np.float64]:
        """
        Calculate value estimations of the model on a list of board states.
        Note: this will crash if serial_states is too large and the network is
        too big. If you get a segmentation fault, call this function on smaller
        chunks of data.
        """

        observations = []
        masks = []
        for serial in serial_states:
            state = game.deserialize_state(serial)
            observations.append(state.observation_tensor())
            masks.append(state.legal_actions_mask())
        values = model.inference(observations, masks)[0]

        return values.flatten()

    return model_value


def get_model_policy_estimator(env: str, config_path: str):
    """ Creates an estimator function that uses a trained model to calculate action policy.
        OpenSpiel MCTS doesn't re-use the search tree, so no problem estimating policies for multiple states.
    """
    game = pyspiel.load_game(env)
    config = load_config(config_path)
    model = load_model_from_checkpoint(config=config, path=config_path, checkpoint_number=10_000)
    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    bot = _init_bot(game, az_evaluator)

    def model_policy(serial_states: list[str], temperature: float) -> list:
        policies = []
        for serial in tqdm(serial_states, desc="Estimating model policy"):
            state = game.deserialize_state(serial)
            root = bot.mcts_search(state)
            policy = np.zeros(game.num_distinct_actions())
            for c in root.children:
                policy[c.action] = c.explore_count
            policy = policy ** (1 / temperature)
            policy /= policy.sum()
            policies.append(policy)
        return policies

    return model_policy

def _init_bot(game, evaluator_):
    """Initializes a bot."""
    return mcts.MCTSBot(
        game,
        2, #c_uct
        300, #rollouts
        evaluator_,
        solve=False,
        dirichlet_noise=None, #assuming evaluation matches
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True)

def get_solver_value_estimator(env: str):
    """
    Create an estimator based on the Connect 4 solver.
    Only supported for Connect 4, could add Pentago.

    :param env: Game environment.
    :return: value_loss function.
    """

    if env != 'connect_four':
        raise NameError('Game name provided not supported: ' + env)

    # Load solver from solver_bot
    game = pyspiel.load_game('connect_four')

    def solver_value(serial_state: str) -> float:
        """
        Calculate ground truth value given by the solver.
        """
        state = game.deserialize_state(serial_state)
        if state.is_terminal():
            # if the state is terminal, the model has nothing to predict.
            raise Exception('Terminal state encountered')
        solver_v, p = connect_four_solver(state, full_info=True)
        return solver_v

    return solver_value


def calculate_solver_values(env: str, serial_states: dict[str, str]):
    """ Calculate and save a database of solver value estimations of serial_states. """
    solver_value = get_solver_value_estimator(env)
    solver_values = {}
    for key, serial in tqdm(serial_states.items(), desc="Estimating solver state values"):
        solver_values[key] = solver_value(serial)

    with open('solver_values.pkl', 'wb') as f:
        pickle.dump(solver_values, f)
