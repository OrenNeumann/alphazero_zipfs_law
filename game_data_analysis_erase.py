import numpy as np
import collections
import re
from tqdm import tqdm
import pyspiel
from alphazero_scaling.solver_bot import connect_four_solver
from alphazero_scaling.AZ_helper_lib import load_model_from_checkpoint, load_config
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import subprocess
from subprocess import PIPE
import pickle

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


def process_games(env, path, max_file_num=39, save_serial=False, save_turn_num=False):
    """ Get game actions from logfiles and use them to
        calculate which board states were played, returning
        a counter of board state frequencies.

        env: name of the game environment.

        path: path to the game files. Should look like 'path/to/file/log-actor'.

        max_file_num: the number of game files.

        save_serial: if True, returns a dict of serialized forms for
        each board visited, used for recreating the state object.
        Each board has many serialized forms, so this
        can't be used as a key, but it's essential to reconstruct the game state.
        This is NOT a good approach for non-Markovian games (like Chess, due to castling
        and 3-rep-draw rules). So actually a bit problematic for Oware and checkers, where
        the number of turns since start/last capture is relevant.

        save_turn_num: if True, returns a dict of the total SUM of the number of turns taken in
        all games. Two dicts are returned, one for turns taken until the state and another
        for turns taken afterwards until the game ended.
        The average num. of turns can be calculated from the sum, by dividing by the count.
        For some games the number of turns is fixed for each state (e.g. Connect 4
        and Pentago, where a stone is added every turn).
        """
    board_counter = collections.Counter()
    serials = dict()
    turns_played = dict()
    turns_to_end = dict()
    if env == 'connect4':
        action_string = r'[xo][0-6]'
    elif env == 'pentago':
        action_string = r'[a-f][1-6][s-z]'
    elif env == 'oware':
        action_string = r'[A-F,a-f]'
    elif env == 'checkers':
        action_string = r'[a-h][1-8][a-h][1-8]'
    else:
        raise NameError('Environment ' + str(env) + ' not supported.')
    # Collect all games from all files
    for i in range(max_file_num):
        file_name = f'/log-actor-{i}.txt'
        print(f'Actor {i}')
        games = _extract_games(path + file_name)
        # Get board positions from all games and add them to counter
        for game in tqdm(games, desc='Processing games'):
            board = _init_board(env)
            actions = re.findall(action_string, game)
            keys = list()
            t=0
            for action in actions:
                _update_board(board, action)
                # Ignore terminal boards:
                if board.is_terminal():
                    break
                key = _board_to_key(board)
                keys.append(key)
                board_counter[key] += 1

                reps = board_counter[key]
                t = board.move_number()
                # Save a representation of each board, for solver estimation later.
                if save_serial and reps == 1:
                    serials[key] = board.serialize()
                # Add number of turns taken (divided later for average).
                # For turns before end, first subtracting current turn, later adding
                # final game length:
                if save_turn_num:
                    turns_played[key] = turns_played.get(key, 0) + t
                    turns_to_end[key] = turns_to_end.get(key, 0) -t
                    #turns_played[key] = (turns_played.get(key, 0) * (reps - 1) + board.move_number()) / reps
            for key in keys:
                turns_to_end[key] += t + 1
    extra_info = {}
    if save_serial:
        extra_info['serials'] = serials
    if save_turn_num:
        # no longer calculating the average, just the sum.
        #turns_played = {x: float(turns_played[x]) / board_counter[x] for x in turns_played}
        #turns_to_end = {x: float(turns_to_end[x]) / board_counter[x] for x in turns_to_end}
        extra_info['turns_played'] = turns_played
        extra_info['turns_to_end'] = turns_to_end
    return board_counter, extra_info


def generate_random_games(env, n=25_000 * 80, save_serial=False):
    """ Quick (and dirty) way to check scaling at T = infinity.
        """
    board_counter = collections.Counter()
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


def get_value_estimators(env: str, config_path: str):
    """
    Create functions that return board-state values. One function for
    trained model estimates, another for solver estimate.
    Only supported for Connect Four, could add Pentago.

    :param env: Game environment.
    :param config_path: Path to the model's config file.
    :return: value_loss function.
    """

    # Load solver from solver_bot
    if env == 'connect4':
        game = CON4

        def solver(s):
            return connect_four_solver(s, full_info=True)
    else:
        raise NameError('Game name provided not supported: ' + env)

    config = load_config(config_path)
    model = load_model_from_checkpoint(config=config, path=config_path, checkpoint_number=10_000)
    evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)

    def value_solver(serial_state: str) -> float:
        """
        Calculate ground truth value given by the solver.
        """
        state = game.deserialize_state(serial_state)
        if state.is_terminal():
            # if the state is terminal, the model has nothing to predict.
            raise Exception('Terminal state encountered')
        solver_v, p = solver(state)
        return solver_v

    def value_model(serial_state: str) -> float:
        """
        Calculate value estimation of the model on a board state.
        """
        state = game.deserialize_state(serial_state)
        if state.is_terminal():
            # if the state is terminal, the model has nothing to predict.
            raise Exception('Terminal state encountered')
        # If I'm not mistaken, an evaluator returns two values for players 0 and 1.
        # The first value is always for player 0 (the player who took the first turn).
        # What the agent actually learns is only to predict the value of player 0 (unlike
        # my old AZ implementation).
        model_v = evaluator.evaluate(state)[state.current_player()]
        return model_v

    return value_solver, value_model


def get_game_length(env: str):
    """ Creates a function calculating the total length of the game
        if played optimally from the current state."""
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


def get_turns(counter, serials, turns, turns_left=True, save_data=True):
    """ Calculate the number of turns for each state, from most to least common.
        If turns_left == True, calculates the number of turns left until the game will end.
            This uses the solver and is extremely slow.
        If turns_left == False, returns the number of turns taken so far. Very fast, and works
            for all games (it's just sorting 'turns').
        """
    if turns_left:
        print('Calculating number of turns till end of game.')
    else:
        print('Getting number of turns taken so far.')
    game_length = get_game_length('connect4')
    n = len(serials)
    y_turns = []
    for entry in tqdm(counter.most_common(), desc='calculating turns'):
        key = entry[0]
        serial = serials[key]
        if turns_left:
            t = game_length(serial) - turns[key]
        else:
            t = turns[key]
        y_turns.append(t)
    y_turns = np.array(y_turns)
    if save_data:
        with open('/mnt/ceph/neumann/zipfs_law/plot_data/turns_left/y_turns.pkl', 'wb') as f:
            pickle.dump(y_turns, f)
    # return cumulative average
    return np.cumsum(y_turns) / (np.arange(n) + 1)

