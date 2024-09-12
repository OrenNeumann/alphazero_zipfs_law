from collections import Counter, defaultdict
import numpy as np
import re
from tqdm import tqdm
import pyspiel
from src.general.general_utils import training_length

"""
Tools for retrieving information from recorded AlphaZero games.
StateCounter is used for extracting game states and frequencies.
"""

ACTION_STRINGS = {
    'connect_four': r'[xo][0-6]',
    'pentago': r'[a-f][1-6][s-z]',
    'oware': r'[A-F,a-f]',
    'checkers': r'[a-h][1-8][a-h][1-8]'
}


class StateCounter(object):
    """ Class for collecting and analyzing game states from AlphaZero logfiles.
        Args:
            env: str, the environment (game).
            save_serial: bool, whether to save the serialized board state, for recreating state objects.
            save_turn_num: bool, whether to save the turn number and turns left for each state.
            save_value: bool, whether to save the value (episode return) of each state.
            cut_early_games: bool, whether to cut out early-training games from the dataset.   

        Methods:
            reset_counters: Reset all counters.
            collect_data: Collect data from logfiles in a directory.
            normalize_counters: Normalize the counters, use after data collection to get the final results.
            prune_low_frequencies: Reduce the size of the data by ignoring low-frequency states.

        Notes:
            Serials: Each board has many serialized forms, so serials can't be used as a key, but they're
            essential to reconstruct the game state. This is NOT a good approach for non-Markovian games 
            (like Chess, due to castling and 3-rep-draw rules). So this approach loses information for 
            Oware and checkers, where the number of turns since start/last capture is relevant.
            
            Turn numbers: For some games the number of turns is fixed for each state (e.g. Connect 4
            and Pentago, where a stone is added every turn). That is not the case for Oware and checkers,
            which is why we average that number.
    """

    def __init__(self,
                 env: str,
                 save_serial: bool = False,
                 save_turn_num: bool = False,
                 save_value: bool = False,
                 cut_early_games: bool = True,
                 cut_extensive: bool = False):
        self.env = env
        self.save_serial = save_serial
        self.save_turn_num = save_turn_num
        self.save_value = save_value
        self.cut_early_games = cut_early_games
        self.cut_extensive = cut_extensive

        self.action_string = ACTION_STRINGS[env]
        self.frequencies = Counter()
        self.serials = dict()
        self.turns_played = defaultdict(int)
        self.turns_to_end = defaultdict(int)
        self.values = dict()
        self.game = pyspiel.load_game(self.env)
        self.normalized = False
        self.draws = 0
        self.games = 0

    def reset_counters(self):
        self.frequencies = Counter()
        self.serials = dict()
        self.turns_played = defaultdict(int)
        self.turns_to_end = defaultdict(int)
        self.values = dict()
        self.normalized = False
        self.draws = 0
        self.games = 0

    def _extract_games(self, file_path: str):
        """ Get the move-list (str) of all games in the file.
            If cut_early_games is True, only include the last 70% of games,
            accounting for short actor files (due to training run crashes).
            If cut_extensive is True, drop the first 80% of games instead. 
            This is a lot more data wasteful."""
        with open(file_path, 'r') as file:
            games = [line.split("Actions: ", 1)[1] for line in file if re.search(r'Game \d+:', line)]
        if self.cut_early_games:
            full_length = training_length(self.env)
            if self.cut_extensive:
                exclude = int(full_length * 0.8)
                games = games[exclude:]
            else:
                include = int(full_length * 0.7)
                games = games[-include:]
        return games

    def collect_data(self, path: str, max_file_num: int, quiet=False, matches=False):
        if self.normalized:
            raise Exception('Data already normalized, reset counters to collect new data.')
        # Collect all games from all files
        for i in range(max_file_num):
            if matches:
                file_name = f'/log-matches-{i}.txt'
            else:
                file_name = f'/log-actor-{i}.txt'
            recorded_games = self._extract_games(path + file_name)
            self.games += len(recorded_games)
            # Get board positions from all games and add them to counter
            for game_record in tqdm(recorded_games, desc=f'Processing actor {i}', disable=quiet):
                board, keys = self._process_game(game_record)
                self._update_frequencies(keys)
                self._update_info_counters(board, keys)

    def _process_game(self, game_record) -> tuple[pyspiel.State, list[str]]:
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
            if self.save_serial and key not in self.serials.keys():
                self.serials[key] = board.serialize()
        # Apply final action (not counted, it's not trained on and it messes up value loss)
        board.apply_action(board.string_to_action(actions[-1]))
        return board, keys

    def _update_frequencies(self, keys: list[str]):
        self.frequencies.update(keys)

    def _update_info_counters(self, board, keys: list[str]):
        """ Update the turn and value counters.
            These counters sum data, divide by count to get average."""
        if self.save_turn_num:
            end = board.move_number() + 1
            for turn, key in enumerate(keys, start=1):
                self.turns_played[key] += turn
                self.turns_to_end[key] += end - turn
        if self.save_value:
            game_return = board.player_return(0)
            if game_return == 0:
                self.draws += 1
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

    def prune_low_frequencies(self, threshold: int) -> None:
        """ Remove all states with a frequency below the threshold.
            Pruning states below threshold=2 roughly reduces the data by an OOM. Pruning
            states below threshold=4 will reduce it by another OOM."""
        self.frequencies = Counter({k: c for k, c in self.frequencies.items() if c >= threshold})
        counters = [self.serials ,self.turns_played, self.turns_to_end, self.values]
        for counter in counters:
            counter = {k: v for k, v in counter.items() if k in self.frequencies.keys()}

    def late_turn_mask(self, threshold: int) -> np.ndarray:
        """ Create mask for late game states when sorting by frequency."""
        if not self.normalized:
            raise Exception('Normalize first.')
        return np.array([self.turns_played[k] >= threshold for k, _ in self.frequencies.most_common()])
    
    def print_draws(self) -> None:
        if not self.save_value:
            raise Exception('No draw data saved.')
        print("Drawn {} out of {} games, ratio: {:.3f}".format(self.draws, self.games, self.draws / self.games))


class RandomGamesCounter(StateCounter):
    """ A StateCounter that collects data from random games, generated on the fly.
        A way to check scaling at T = infinity.
        """

    def __init__(self,
                 env: str,
                 save_serial=False,
                 save_turn_num=False,
                 save_value=False):
        super().__init__(env=env,
                         save_serial=save_serial,
                         save_turn_num=save_turn_num,
                         save_value=save_value)

    def collect_data(self, n: int = 25_000 * 80, *args, **kwargs):
        if self.normalized:
            raise Exception('Data already normalized, reset counters to collect new data.')
        for episode in tqdm(range(n), desc='Generating games'):
            board, keys = self._process_game()
            self._update_frequencies(keys)
            self._update_info_counters(board, keys)

    def _process_game(self, *args, **kwargs):
        board = self.game.new_initial_state()
        keys = list()
        while not board.is_terminal():
            board.apply_action(np.random.choice(board.legal_actions()))
            if board.is_terminal():
                break
            key = str(board)
            keys.append(key)
            if self.save_serial and key not in self.serials.keys():
                self.serials[key] = board.serialize()
        return board, keys
