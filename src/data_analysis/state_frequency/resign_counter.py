
"""
Count states if player resignation is implemented.
Note: this affects 'turns_to_end', which counts the distance to resignation instead of 
distance to terminal state.
"""
import re
import numpy as np
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.alphazero_scaling.loading import load_model_from_checkpoint, load_config


def get_model(path):
    config = load_config(path)
    return load_model_from_checkpoint(config=config, path=path, checkpoint_number=10_000)

class ResignationCounter(StateCounter):
    def __init__(self, model,**kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.v_resign = -0.8
        self.total_games = 0
        self.resign_rate = 0
        self.average_resign_turn = 0

    def _reset(self):
        self.total_games = 0
        self.resign_rate = 0
        self.average_resign_turn = 0

    def collect_data(self, path, max_file_num):
        self._reset()
        if self.normalized:
            raise Exception('Data already normalized, reset counters to collect new data.')
        # Collect all games from all files
        for i in range(max_file_num):
            file_name = f'/log-actor-{i}.txt'
            recorded_games = self._extract_games(path + file_name)
            self.total_games += len(recorded_games)
            # Get board positions from all games and add them to counter
            for game_record in tqdm(recorded_games, desc=f'Processing actor {i}'):
                board, keys = self._process_game(game_record)
                self._update_frequencies(keys)
                self._update_info_counters(board, keys)
        print(f'Resigned {self.resign_rate} out of {self.total_games} games, percentage: {self.resign_rate/self.total_games*100:.2f}%')
        print(f'Average turn to resign: {self.average_resign_turn/self.resign_rate:.2f}')

    def _process_game(self, game_record):
        board = self.game.new_initial_state()
        actions = re.findall(self.action_string, game_record)
        boards = list()
        for action in actions[:-1]:
            board.apply_action(board.string_to_action(action))
            boards.append(board.clone())     
        board.apply_action(board.string_to_action(actions[-1]))
        if not board.is_terminal():
            raise Exception('Game ended prematurely. Maybe a corrupted file?')
        
        decisions = self._should_resign([board])
        keys = list()
        for board, resigned in zip(boards, decisions):
            if resigned:
                self.resign_rate += 1
                self.average_resign_turn += len(keys)
                break
            key = str(board)
            keys.append(key)
            if self.save_serial and key not in self.serials.keys():
                self.serials[key] = board.serialize()
        return board, keys
    
    def _should_resign(self, boards):
        """ Return a list of decisions to resign or not for each board. """
        obs = []
        masks = []
        for board in boards:
            obs.append(board.observation_tensor())
            masks.append(board.legal_actions_mask())

        values = self.model.inference(obs, masks)[0]
        values *= np.array([(-1)**n for n in range(len(boards))])
        return values <= self.v_resign
