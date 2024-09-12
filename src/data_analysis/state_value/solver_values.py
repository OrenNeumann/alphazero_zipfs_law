import numpy as np
import subprocess
from subprocess import PIPE
import pyspiel

GAME = pyspiel.load_game('connect_four')

def connect_four_solver(states: list) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ Produces an optimal-scores vector for a game state.
        Takes a list of states and returns for each a mask of legal moves, plus a vector
        of scores for each move. The scores are integers indicating how early the game will end, with 
        a +/- sign for win/loss. Higher score = better move.

        This code uses the open source solver available in:
        https://connect4.gamesolver.org/

        In order to use this, download the Github repo:
        https://github.com/PascalPons/connect4
        Hit 'make' and copy the call_solver.sh script into the folder.
        Then download the openings book here: (7x6.book)
        https://github.com/PascalPons/connect4/releases/tag/book
        and place it in the working directory.
        """
    procs = []
    masks = []
    for state in states:
        # The solver starts counting moves from 1:
        moves = ''.join([str(action.action + 1) for action in state.full_history()])
        masks.append(state.legal_actions_mask())
        procs.append(subprocess.Popen(["src/alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE))
    scores = []
    for proc in procs:
        out, _ = proc.communicate()
        out = out.split()
        if moves == "":
            scores.append(np.array(out, dtype=int))
        else:  # ignore 1st output (=moves):
            scores.append(np.array(out[1:], dtype=int))
    return masks, scores


def _get_values(states: list):
    masks, scores = connect_four_solver(states)
    values = []
    for i, score_vec in enumerate(scores):
        mask = masks[i]
        p = np.extract(mask, score_vec)
        # v=[1,0,-1] for win, draw, loss
        values.append(max(np.sign(p)))
    return values


def _get_optimal_moves(states: list):
    _, scores = connect_four_solver(states)
    optimal_moves = [np.sign(score_vec) == max(np.sign(score_vec)) for score_vec in scores]
    return optimal_moves


def _get_optimal_policy(states: list):
    masks, scores = connect_four_solver(states)
    values = []
    probs = []
    for i, score_vec in enumerate(scores):
        mask = masks[i]
        p = np.extract(mask, score_vec)
        values.append(max(np.sign(p)))
        if (p > 0).any():  # Win
            p[p < np.amax(p)] = 0
            p[p != 0] = 1
        elif (p == 0).any():  # Draw
            p = (p == 0).astype(int)
        else:  # Loss
            p[p < np.amax(p)] = 0
            p[p != 0] = 1
        probs.append(p / sum(p))
    return values, probs

def solver_values(serial_states: list[str]) -> list[float]:
    """
    Calculate ground truth value given by the solver.
    Returns the value for the first player.
    """
    states = [GAME.deserialize_state(serial) for serial in serial_states]
    solver_v = _get_values(states)
    player_0_values = [solver_v[i] * (1 - 2 * state.current_player()) for i, state in enumerate(states)]
    return player_0_values


def solver_optimal_moves(serial_state: list[str]) -> list[np.ndarray]:
    """
    Returns a mask of optimal moves (all moves that lead to the best possible game outcome).
    If no win/draw is possible, returns True for all moves (including illegal moves!)
    """
    states = [GAME.deserialize_state(serial) for serial in serial_state]
    return _get_optimal_moves(states)
