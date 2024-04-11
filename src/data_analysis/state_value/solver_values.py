import numpy as np
import subprocess
from subprocess import PIPE
import pyspiel
from tqdm import tqdm

GAME = pyspiel.load_game('connect_four')

def connect_four_solver(states: list, return_policy=False):
    """ Produces an optimal policy for a game state.
        Takes a list of states and returns their values, or values and policies.

        This code uses the open source solver available in:
        https://connect4.gamesolver.org/

        In order to use this, download the Github repo:
        https://github.com/PascalPons/connect4
        Hit 'make' and copy the call_solver.sh script into the folder.
        Then download the openings book here: (7x6.book)
        https://github.com/PascalPons/connect4/releases/tag/book
        and place it in the top directory.
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
    
    if return_policy:
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

    values = []
    for i, score_vec in enumerate(scores):
        mask = masks[i]
        p = np.extract(mask, score_vec)
        # v=[1,0,-1] for win, draw, loss
        values.append(max(np.sign(p)))
    return values


def solver_values(serial_state: list[str]) -> list[float]:
    """
    Calculate ground truth value given by the solver.
    Returns the value for the first player.
    """
    states = [GAME.deserialize_state(serial) for serial in serial_state]
    solver_v = connect_four_solver(states)
    player_0_values = [solver_v[i] * (1 - 2 * state.current_player()) for i, state in enumerate(states)]
    return player_0_values
