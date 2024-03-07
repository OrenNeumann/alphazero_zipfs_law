import numpy as np
from collections import Counter, deque
import re
from tqdm import tqdm
import pyspiel
from random import sample
from src.data_analysis.state_frequency.state_counter import StateCounter


class BufferCounter(StateCounter):
    def __init__(self,
                 env: str,
                 save_serial=False,
                 save_turn_num=False,
                 save_value=False,
                 sample_unique_states=False):
        super().__init__(env, save_serial, save_turn_num, save_value)
        self.sample_unique_states = sample_unique_states
        self.batch_size = 2 ** 10
        self.buffer_size = 2 ** 16
        self.buffer_reuse = 10
        self.sample_threshold = int(self.buffer_size / self.buffer_reuse)
        self.buffer = deque(maxlen=self.buffer_size)




def process_games_with_buffer(env: str,
                              path: str,
                              sample_unique_states=False,
                              max_file_num: int = 39) -> tuple[Counter, dict]:
    """ Same as process_games, but the state counts are the number of times
        an agent will see each state when training with prioritized experience replay."""
    batch_size = 2 ** 10
    buffer_size = 2 ** 16
    buffer_reuse = 10
    sample_threshold = int(buffer_size / buffer_reuse)
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
                    if sample_unique_states:
                        samples = _sample_unique_states(buffer, batch_size)
                    else:
                        samples = _sample_uniformly(buffer, batch_size)
                    for k in samples:
                        board_counter[k] += 1
                        if board_counter[k] == 1:
                            serials[k] = board.serialize()

            if not board.is_terminal():
                raise Exception('Game ended prematurely. Maybe a corrupted file?')

    extra_info = {'serials': serials}

    return board_counter, extra_info


def _sample_unique_states(buffer, batch_size):
    unique_keys = list(set(buffer))
    if len(unique_keys) < batch_size:
        raise Exception(f'Batch size ({batch_size}) larger than number of unique keys ({len(unique_keys)}).')
    return sample(unique_keys, batch_size)


def _sample_uniformly(buffer, batch_size):
    return sample(buffer, batch_size)
