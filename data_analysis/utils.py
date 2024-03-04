import collections
from alphazero_scaling.solver_bot import connect_four_solver
from alphazero_scaling.loading import load_model_from_checkpoint, load_config
import subprocess
from subprocess import PIPE
import pickle
import numpy as np
import numpy.typing as npt
import pyspiel
from tqdm import tqdm
import yaml


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
        return pyspiel.load_game('connect_four')
    elif env == 'pentago':
        return pyspiel.load_game('pentago')
    elif env == 'oware':
        return pyspiel.load_game('oware')
    elif env == 'checkers':
        return pyspiel.load_game('checkers')
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

    def model_value(serial_states: list[str]) -> npt.NDArray[np.float64]:
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


def _get_solver_game_length(env: str):
    """ Creates a function calculating the total length of the game
        if played optimally from the current state.
        uses the solver."""
    if env == 'connect4':
        game = pyspiel.load_game('connect_four')
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


def calculate_solver_values(env: str, serial_states):
    """ Calculate and save a database of solver value estimations of serial_states. """
    solver_value = get_solver_value_estimator(env)
    solver_values = {}
    for key, serial in tqdm(serial_states.items(), desc="Estimating solver state values"):
        solver_values[key] = solver_value(serial)

    with open('solver_values.pkl', 'wb') as f:
        pickle.dump(solver_values, f)


def models_path():
    with open("config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['paths']['models_dir']


def fit_power_law(freq,
                  up_bound,
                  low_bound,
                  full_equation=False,
                  name='',
                  n_points=2 * 10 ** 6):
    # Fit a power-law
    x_nums = np.arange(up_bound)[low_bound:] + 1
    y_nums = freq[low_bound:up_bound]
    [m, c] = np.polyfit(np.log10(x_nums), np.log10(y_nums), deg=1, w=2 / x_nums)
    exp = str(round(-m, 2))
    if full_equation:
        const_exp = str(round(-c / m, 2))
        equation = r'$ \left( \frac{n}{ 10^{'+const_exp+r'} } \right) ^ {\mathbf{'+exp+r'}}$'
    else:
        if name == '':
            equation = r'$\alpha = ' + exp + '$'
        else:
            equation = name + r', $\alpha = ' + exp + '$'

    x_fit = np.array([1, n_points])
    y_fit = 10 ** c * x_fit ** m
    return x_fit, y_fit, equation


def fit_logaritm(freq,
                  up_bound,
                  low_bound,
                  n_points=2 * 10 ** 6):
    # Fit a power-law
    x_nums = np.arange(up_bound)[low_bound:] + 1
    y_nums = freq[low_bound:up_bound]
    [m, c] = np.polyfit(np.log10(x_nums), y_nums, deg=1, w=2 / x_nums)
    slope = str(round(m, 2))
    const = str(round(c, 2))
    equation = r'$ {'+slope+r'} \cdot \log_{10} n + {'+const+r'}$'

    x_fit = np.array([1, n_points])
    y_fit = m * np.log10(x_fit) + c
    return x_fit, y_fit, equation
