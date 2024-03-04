import numpy as np
from collections import Counter, deque
import re
from tqdm import tqdm
import pyspiel
from random import sample

"""
Functions for retrieving information from recorded AlphaZero games.
'process_games' is used for extracting game states and frequencies.

Note: This code assumes the logfiles only contain legal moves, 
    be careful using it on another dataset.
"""

CON4 = pyspiel.load_game('connect_four')
PENT = pyspiel.load_game('pentago')
OWAR = pyspiel.load_game('oware')
CHEC = pyspiel.load_game('checkers')


def _init_board(env):
    """Return a pyspiel state"""
    if env == 'connect4':
        return CON4.new_initial_state()
    elif env == 'pentago':
        return PENT.new_initial_state()
    elif env == 'oware':
        return OWAR.new_initial_state()
    elif env == 'checkers':
        return CHEC.new_initial_state()
    else:
        raise NameError('Environment ' + str(env) + ' not supported.')

def _get_action_string(env: str):
    """ Return the string format used to encode actions, as a 
        regular expression.
    """
    if env == 'connect4':
        actions = r'[xo][0-6]'
    elif env == 'pentago':
        actions = r'[a-f][1-6][s-z]'
    elif env == 'oware':
        actions = r'[A-F,a-f]'
    elif env == 'checkers':
        actions = r'[a-h][1-8][a-h][1-8]'
    else:
        raise NameError('Environment ' + str(env) + ' not supported.')
    return actions

def _update_board(board, action):
    board.apply_action(board.string_to_action(action))


def _board_to_key(board):
    # Use pyspiel's string as key.
    # Using str rather than observation, since asking
    # for an observation of a final board throw an error.
    return str(board)


def _extract_games(file_name):
    """ Get the move-list (str) of all games in the file."""
    with open(file_name, 'r') as file:
        games = [line.split("Actions: ", 1)[1] for line in file if re.search(r'Game \d+:', line)]
    return games


def process_games(env: str, path: str, max_file_num: int = 39, save_serial: bool = False,
                  save_turn_num: bool = False, save_value: bool = False) -> tuple[Counter, dict]:
    """ Get game actions from logfiles and use them to calculate which board states were
        played, returning a counter of board state frequencies.
        For convenience, the first (empty) board and terminal boards are not counted.
        The empty board has well known frequency, so is a waste of time to compute.
        The terminal boards mess up value analysis (AZ evaluators can't evaluate them)
        and are not part of the training process.

        :param env: game environment.

        :param path: path to the game files. Should look like 'path/to/file/log-actor'.

        :param max_file_num: number of game files to scrape.

        :param save_serial: if True, returns a dict of serialized forms for each board visited,
        used for recreating the state object. Each board has many serialized forms, so this
        can't be used as a key, but it's essential to reconstruct the game state.
        This is NOT a good approach for non-Markovian games (like Chess, due to castling
        and 3-rep-draw rules). So actually a bit problematic for Oware and checkers, where
        the number of turns since start/last capture is relevant.

        :param save_turn_num: if True, returns a dict of the total SUM of the number of turns taken in
        all games. Two dicts are returned, one for turns taken until the state and another
        for turns taken afterwards until the game ended.
        The average num. of turns can be calculated from the sum, by dividing by the count.
        For some games the number of turns is fixed for each state (e.g. Connect 4
        and Pentago, where a stone is added every turn).

        :param save_value: if True, returns a dict of total sum of values for each state. Value is the
        return of the game the state appeared in, one of [1,0,-1]. Divide this sum by the count
        to get the average value of a state.
        """
    board_counter = Counter()
    serials = dict()
    turns_played = dict()
    turns_to_end = dict()
    values = dict()
    action_string = _get_action_string(env)

    # Collect all games from all files
    for i in range(max_file_num):
        file_name = f'/log-actor-{i}.txt'
        games = _extract_games(path + file_name)
        # Get board positions from all games and add them to counter
        for game in tqdm(games, desc=f'Processing actor {i}'):
            board = _init_board(env)
            actions = re.findall(action_string, game)
            keys = list()
            for action in actions:
                _update_board(board, action)
                # Don't count terminal states (not part of training, mess up value loss)
                if board.is_terminal():
                    break
                key = _board_to_key(board)
                keys.append(key)
                board_counter[key] += 1

                if save_serial and board_counter[key] == 1:
                    serials[key] = board.serialize()

            if not board.is_terminal():
                raise Exception('Game ended prematurely. Maybe a corrupted file?')

            # turn and value data is summed up, divide the sum by the count to get an average.
            if save_turn_num:
                end = board.move_number() + 1
                for turn, key in enumerate(keys, start=1):
                    turns_played[key] = turns_played.get(key, 0) + turn
                    turns_to_end[key] = turns_to_end.get(key, 0) + end - turn
            if save_value:
                for turn, key in enumerate(keys, start=1):
                    values[key] = values.get(key, 0) + board.player_return(0)

    extra_info = {}
    if save_serial:
        extra_info['serials'] = serials
    if save_turn_num:
        extra_info['turns_played'] = turns_played
        extra_info['turns_to_end'] = turns_to_end
    if save_value:
        extra_info['values'] = values
    return board_counter, extra_info


def generate_random_games(env, n=25_000 * 80, save_serial=False):
    """ Quick (and dirty) way to check scaling at T = infinity.
        """
    board_counter = Counter()
    serials = dict()
    # Collect all games from all files

    for game in tqdm(range(n), desc='Generating games'):
        # Initialize the board
        board = _init_board(env)
        while not board.is_terminal():
            board.apply_action(np.random.choice(board.legal_actions()))
            board_key = _board_to_key(board)
            board_counter[board_key] += 1
            # Save a representation of each board, for solver estimation later.
            if save_serial and board_counter[board_key] == 1:
                serials[board_key] = board.serialize()
    if save_serial:
        return board_counter, serials
    else:
        return board_counter


def process_games_with_buffer(env: str, path: str, max_file_num: int = 39) -> tuple[Counter, dict]:
    """ Same as process_games, but the state counts are the number of times
        an agent will see each state when training with prioritized experience replay."""
    batch_size = 2**10
    buffer_size = 2**16
    buffer_reuse = 10
    sample_threshold = int(buffer_size/buffer_reuse)
    buffer = deque(maxlen=buffer_size)
    new_states = 0

    board_counter = Counter()
    serials = dict()
    action_string = _get_action_string(env)
    # Collect all games from all files
    for i in range(max_file_num):
        file_name = f'/log-actor-{i}.txt'
        games = _extract_games(path + file_name)
        # Get board positions from all games and add them to counter
        for game in tqdm(games, desc=f'Processing actor {i}'):
            board = _init_board(env)
            actions = re.findall(action_string, game)
            for action in actions:
                _update_board(board, action)
                # Don't count terminal states (not part of training, mess up value loss)
                if board.is_terminal():
                    break
                key = _board_to_key(board)
                buffer.append(key)
                new_states += 1

                if new_states == sample_threshold:
                    new_states = 0
                    samples = _sample_from_buffer(buffer,batch_size)
                    for k in samples:
                        board_counter[k] += 1
                        if board_counter[k] == 1:
                            serials[k] = board.serialize()

            if not board.is_terminal():
                raise Exception('Game ended prematurely. Maybe a corrupted file?')

    extra_info = {}
    extra_info['serials'] = serials

    return board_counter, extra_info


def _sample_from_buffer(buffer, batch_size):
    unique_keys = list(set(buffer))
    if len(unique_keys) < batch_size:
        raise Exception(f'Batch size ({batch_size}) larger than number of unique keys ({len(unique_keys)}).')
    return sample(unique_keys, batch_size)
    