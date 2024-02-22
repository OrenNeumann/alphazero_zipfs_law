import collections

from alphazero_scaling.solver_bot import connect_four_solver
from alphazero_scaling.AZ_helper_lib import load_model_from_checkpoint, load_config
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import subprocess
from subprocess import PIPE
import pickle
import numpy as np
import numpy.typing as npt
import pyspiel
from tqdm import tqdm


CON4 = pyspiel.load_game('connect_four')
PENT = pyspiel.load_game('pentago')
OWAR = pyspiel.load_game('oware')
CHEC = pyspiel.load_game('checkers')


def sort_by_frequency(data: dict, counter: collections.Counter) -> np.array:
    """ Sort any state-data by descending order of frequency in counter."""
    sorted_data = np.zeros(len(counter))
    for idx, entry in enumerate(tqdm(counter.most_common())):
        key = entry[0]
        sorted_data[idx] = data[key]
    return sorted_data


def _get_game(env):
    """Return a pyspiel game"""
    if env == 'connect4':
        return CON4
    elif env == 'pentago':
        return PENT
    elif env == 'oware':
        return OWAR
    elif env == 'checkers':
        return CHEC
    else:
        raise NameError('Environment ' + str(env) + ' not supported.')


def get_model_value_estimator(env: str, config_path: str):
    """ Creates an estimator function that uses a trained model to calculate state values.

        :param env: Game environment.
        :param config_path: Path to trained model.
    """
    game = _get_game(env)
    config = load_config(config_path)
    model = load_model_from_checkpoint(config=config, path=config_path, checkpoint_number=10_000)

    def model_value(serial_states: list[str]) -> npt.NDArray[float]:
        """
        Calculate value estimations of the model on a list of board states.
        Note: this will crash if serial_states is too large and the network is
        too big. If you get a segmentation fault, call this function on smaller
        chunks of data.
        """

        observations = []
        masks = []
        for i, serial in enumerate(serial_states):
            state = game.deserialize_state(serial)
            observations.append(state.observation_tensor())
            masks.append(state.legal_actions_mask())
        values = model.inference(observations, masks)[0]

        return values.flatten()

    return model_value


def get_solver_value_estimator(env: str):
    """
    Create an estimator based on the Connect 4 solver.
    Only supported for Connect 4, could add Pentago.

    :param env: Game environment.
    :return: value_loss function.
    """

    if env != 'connect4':
        raise NameError('Game name provided not supported: ' + env)

    # Load solver from solver_bot
    game = CON4

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


def _get_solver_game_length(env: str):
    """ Creates a function calculating the total length of the game
        if played optimally from the current state.
        uses the solver."""
    if env == 'connect4':
        game = CON4
    else:
        raise NameError('Game name provided not supported: ' + env)

    def game_length(serial_state: str):
        state = game.deserialize_state(serial_state)
        moves = ''
        for action in state.full_history():
            # The solver starts counting moves from 1:
            moves += str(action.action + 1)

        # Call the solver and get an array of scores for all moves.
        # Optimal legal moves will have the highest score.
        # out = subprocess.run(["./connect4-master/call_solver.sh", moves], capture_output=True)
        out = subprocess.run(["alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE)
        out = out.stdout.split()
        if moves == "":
            scores = np.array(out, dtype=int)
        else:  # ignore 1st output (=moves):
            scores = np.array(out[1:], dtype=int)

        distance_to_end = abs(max(scores))
        return 42 - distance_to_end

    return game_length


def get_solver_turns(counter, serials, turns, save_data=True):
    """ Calculate the number of turns left until the game will end, for each state, from most to least common.
        This uses the solver and is extremely slow.
        """
    print('Calculating number of turns till end of game (with solver).')
    game_length = _get_solver_game_length('connect4')
    n = len(serials)
    y_turns = []
    for entry in tqdm(counter.most_common()):
        key = entry[0]
        serial = serials[key]
        t = game_length(serial) - turns[key]
        y_turns.append(t)
    y_turns = np.array(y_turns)
    if save_data:
        with open('/mnt/ceph/neumann/zipfs_law/plot_data/turns_left/y_turns.pkl', 'wb') as f:
            pickle.dump(y_turns, f)
    # return cumulative average
    return np.cumsum(y_turns) / (np.arange(n) + 1)
