from collections import Counter
import re
from tqdm import tqdm
import pyspiel
from src.general.general_utils import training_length

"""
Tools for retrieving information from recorded AlphaZero games.
StateCounter is used for extracting game states and frequencies.

Note: This code assumes the logfiles only contain legal moves, 
    be careful using it on another dataset.
"""





class StateCounter:
    """ Class for collecting and analyzing game states from AlphaZero logfiles.
    """
    def __init__(self,
                 env: str,
                 save_serial=False,
                 save_turn_num=False,
                 save_value=False,
                 cut_early_games=True):
        self.env = env
        self.save_serial = save_serial
        self.save_turn_num = save_turn_num
        self.save_value = save_value
        self.cut_early_games = cut_early_games

        self.action_formats = {'connect_four': r'[xo][0-6]',
                               'pentago': r'[a-f][1-6][s-z]',
                               'oware': r'[A-F,a-f]',
                               'checkers': r'[a-h][1-8][a-h][1-8]'}

        self.action_string = self.action_formats[env]

        #self.action_string = action_string(self.env)
        self.frequencies = Counter()
        self.serials = dict()
        self.turns_played = dict()
        self.turns_to_end = dict()
        self.values = dict()
        self.game = pyspiel.load_game(self.env)
        self.normalized = False

    def reset_counters(self):
        self.frequencies = Counter()
        self.serials = dict()
        self.turns_played = dict()
        self.turns_to_end = dict()
        self.values = dict()
        self.normalized = False

    def _extract_games(self, file_path):
        """ Get the move-list (str) of all games in the file.
            If cut_early_games is True, only include the last 70% of games,
            accounting for short actor files (due to training run crashes)."""
        with open(file_path, 'r') as file:
            games = [line.split("Actions: ", 1)[1] for line in file if re.search(r'Game \d+:', line)]
        if self.cut_early_games:
            full_length = training_length(self.env)
            include = int(full_length * 0.7)
            games = games[-include:]
        return games

    def collect_data(self, path, max_file_num):
        if self.normalized:
            raise Exception('Data already normalized, reset counters to collect new data.')
        # Collect all games from all files
        for i in range(max_file_num):
            file_name = f'/log-actor-{i}.txt'
            recorded_games = self._extract_games(path + file_name)
            # Get board positions from all games and add them to counter
            for game_record in tqdm(recorded_games, desc=f'Processing actor {i}'):
                board, keys = self._process_game(game_record)
                self._update_info_counters(board, keys)

    def _process_game(self, game_record):
        """ Process a single game.
            Update the board counter and return the final board + keys of board positions played.
        """
        board = self.game.new_initial_state()
        actions = re.findall(self.action_string, game_record)
        keys = list()
        for action in actions[:-1]:
            board.apply_action(board.string_to_action(action))
            # Using pyspiel's string as key rather than observation, since asking
            # for an observation of a final board throws an error.
            key = str(board)
            keys.append(key)
            self.frequencies[key] += 1

            if self.save_serial and self.frequencies[key] == 1:
                self.serials[key] = board.serialize()
        # Apply final action (not counted, it's not trained on and it messes up value loss)
        board.apply_action(board.string_to_action(actions[-1]))
        if not board.is_terminal():
            raise Exception('Game ended prematurely. Maybe a corrupted file?')
        return board, keys

    def _update_info_counters(self, board, keys):
        """ Update the turn and value counters.
            These counters sum data, divide by count to get average."""
        if self.save_turn_num:
            end = board.move_number() + 1
            for turn, key in enumerate(keys, start=1):
                self.turns_played[key] = self.turns_played.get(key, 0) + turn
                self.turns_to_end[key] = self.turns_to_end.get(key, 0) + end - turn
        if self.save_value:
            game_return = board.player_return(0)
            for key in keys:
                self.values[key] = self.values.get(key, 0) + game_return

    def normalize_counters(self):
        """ Normalize sum counters, dividing the sum by the count for all variables that aggregate sums.
            Currently modifies the original counters rather than create copies (to save space)."""
        if self.normalized:
            raise Exception('Data already normalized.')
        counters = []
        if self.save_turn_num:
            counters.append(self.turns_played)
            counters.append(self.turns_to_end)
        if self.save_value:
            counters.append(self.values)
        for counter in counters:
            for key, count in self.frequencies.items():
                counter[key] /= count
        self.normalized = True

    def prune_low_frequencies(self, threshold):
        """ Remove all states with a frequency below the threshold.
            Pruning states below threshold=2 roughly reduces the data by an OOM. Pruning
            states below threshold=4 will reduce it by another OOM."""
        self.frequencies = Counter({k: c for k, c in self.frequencies.items() if c >= threshold})
