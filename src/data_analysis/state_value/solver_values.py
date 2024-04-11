import numpy as np
import subprocess
from subprocess import PIPE
import pyspiel


def connect_four_solver(state, return_policy=False):
    """ Produces an optimal policy for the current state.

        This code uses the open source solver available in:
        https://connect4.gamesolver.org/

        In order to use this, download the Github repo:
        https://github.com/PascalPons/connect4
        Hit 'make' and copy the call_solver.sh script into the folder.
        Then download the openings book here: (7x6.book)
        https://github.com/PascalPons/connect4/releases/tag/book
        and place it in the top directory.
        """

    # The solver starts counting moves from 1:
    moves = ''.join([str(action.action + 1) for action in state.full_history()])

    # Call the solver and get an array of scores for all moves.
    # Optimal legal moves will have the highest score.
    out = subprocess.run(["src/alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE)
    out = out.stdout.split()
    if moves == "":
        scores = np.array(out, dtype=int)
    else:  # ignore 1st output (=moves):
        scores = np.array(out[1:], dtype=int)

    mask = state.legal_actions_mask()
    p = np.extract(mask, scores)
    # v=[1,0,-1] for win, draw, loss
    v = max(np.sign(p))
    if return_policy:
        if (p > 0).any():  # Win
            p[p < np.amax(p)] = 0
            p[p != 0] = 1
        elif (p == 0).any():  # Draw
            p = (p == 0).astype(int)
        else:  # Loss
            p[p < np.amax(p)] = 0
            p[p != 0] = 1
        return v, p / sum(p)
    return v


def get_solver_value_estimator():
    """
    Create an estimator based on the Connect 4 solver.
    Only supported for Connect 4, could add Pentago.
    """
    game = pyspiel.load_game('connect_four')

    def solver_value(serial_state: str) -> float:
        """
        Calculate ground truth value given by the solver.
        Returns the value for the current player.
        """
        state = game.deserialize_state(serial_state)
        """
        if state.is_terminal():
            # if the state is terminal, the model has nothing to predict.
            raise Exception('Terminal state encountered')"""
        solver_v = connect_four_solver(state)
        return solver_v * (1 - 2 * state.current_player())

    return solver_value
