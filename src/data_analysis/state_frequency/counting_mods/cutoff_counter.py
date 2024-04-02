import re
from src.data_analysis.state_frequency.state_counter import StateCounter


class CutoffCounter(StateCounter):
    """
    Count states when using late-state cutoff.
    'Cutoff' is the number of turns before the end of the game to stop counting states.
    """
    def __init__(self, cutoff=80, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff

    def _process_game(self, game_record):
        """ Process a single game, counting states only until the cutoff point.
        """
        board = self.game.new_initial_state()
        actions = re.findall(self.action_string, game_record)
        keys = list()
        for action in actions[:self.cutoff]:
            board.apply_action(board.string_to_action(action))
            key = str(board)
            keys.append(key)
            if self.save_serial and key not in self.serials.keys():
                self.serials[key] = board.serialize()
        for action in actions[self.cutoff:]:
            board.apply_action(board.string_to_action(action))
        return board, keys
