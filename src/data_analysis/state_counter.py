import numpy as np
from collections import Counter, deque
import re
from tqdm import tqdm
import pyspiel
from random import sample
from src.general.general_utils import action_string


class StateCounter:

    def __init__(self,
                 env: str,
                 save_serial: bool = False,
                 save_turn_num: bool = False,
                 save_value: bool = False
                 ):
        if env == 'connect4':
            env = 'connect_four'
        self.env = env
        self.save_serial = save_serial
        self.save_turn_num = save_turn_num
        self.save_value = save_value

        self.board_counter = Counter()
        self.serials = dict()
        self.turns_played = dict()
        self.turns_to_end = dict()
        self.values = dict()
        """"
        self.action_formats = {'connect4': r'[xo][0-6]',
                               'pentago': r'[a-f][1-6][s-z]',
                               'oware': r'[A-F,a-f]',
                               'checkers': r'[a-h][1-8][a-h][1-8]'}
        """
        self.action_string = action_string(self.env)

    def collect_data(self, path, max_file_num):
        # Collect all games from all files
        for i in range(max_file_num):
            file_name = f'/log-actor-{i}.txt'
            games = _extract_games(path + file_name)
            # Get board positions from all games and add them to counter
            for game in tqdm(games, desc=f'Processing actor {i}'):
                board, keys = self._process_game(game)
                self._update_info_counters(board, keys)

    def _process_game(self, game):
        """Process a single game.
        Update the board counter and return the fnal board + keys of board positions played.
        """
        board = pyspiel.load_game(game).new_initial_state()
        actions = re.findall(self.action_string, game)
        keys = list()
        for action in actions[-1]:
            board.apply_action(board.string_to_action(action))
            # Using pyspiel's string as key rather than observation, since asking
            # for an observation of a final board throws an error.
            key = str(board)
            keys.append(key)
            self.board_counter[key] += 1

            if self.save_serial and self.board_counter[key] == 1:
                self.serials[key] = board.serialize()
        # Apply final action (not counted, it's not trained on and it messes up value loss)
        board.apply_action(board.string_to_action(actions[-1]))
        if not board.is_terminal():
            raise Exception('Game ended prematurely. Maybe a corrupted file?')
        return board, keys

    def _update_info_counters(self, board, keys):
        # turn and value data is summed up, divide the sum by the count to get an average.
        if self.save_turn_num:
            end = board.move_number() + 1
            for turn, key in enumerate(keys, start=1):
                self.turns_played[key] = self.turns_played.get(key, 0) + turn
                self.turns_to_end[key] = self.turns_to_end.get(key, 0) + end - turn
        if self.save_value:
            game_return = board.player_return(0)
            for turn, key in enumerate(keys, start=1):
                self.values[key] = self.values.get(key, 0) + game_return


def _extract_games(file_name):
    """ Get the move-list (str) of all games in the file."""
    with open(file_name, 'r') as file:
        games = [line.split("Actions: ", 1)[1] for line in file if re.search(r'Game \d+:', line)]
    return games
