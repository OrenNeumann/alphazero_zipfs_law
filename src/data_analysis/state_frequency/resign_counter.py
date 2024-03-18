
"""
Count states if player resignation is implemented.
"""
import re
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
            if self.save_serial and key not in self.serials.keys():
                self.serials[key] = board.serialize()

            if self._should_resign(board):
                break
        return board, keys
    
    def _should_resign(self, board):
        obs = board.observation_tensor()
        mask = board.legal_actions_mask()
        value = self.model.inference([obs], [mask])[0][0][0] 
        if board.current_player() == 1:
            value = -value
        if value <= self.v_resign:
            return True
        else:
            return False