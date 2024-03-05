import collections
from src.alphazero_scaling.solver_bot import connect_four_solver
from src.alphazero_scaling.loading import load_model_from_checkpoint, load_config
import subprocess
from subprocess import PIPE
import pickle
import numpy as np
import numpy.typing as npt
import pyspiel
from tqdm import tqdm


def sort_by_frequency(data: dict, counter: collections.Counter) -> np.array:
    """ Sort any state-data by descending order of frequency in counter."""
    sorted_data = np.zeros(len(counter))
    for idx, entry in enumerate(tqdm(counter.most_common())):
        key = entry[0]
        sorted_data[idx] = data[key]
    return sorted_data



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
        out = subprocess.run(["src/alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE)
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