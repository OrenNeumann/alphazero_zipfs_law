

import numpy as np
import subprocess
from subprocess import PIPE



def connect_four_solver(state):
    """ Produces an optimal policy for the current state.

        This code uses the open source solver available in:
        https://connect4.gamesolver.org/

        In order to use this, download the Github repo:
        https://github.com/PascalPons/connect4
        Hit 'make' and copy the call_solver.sh script into the folder.
        Then download the openings book here: (7x6.book)
        https://github.com/PascalPons/connect4/releases/tag/book
        and place it in the parent directory.
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
    if (p > 0).any():  # Win
        v = 1
        p[p < np.amax(p)] = 0
        p[p != 0] = 1
    elif (p == 0).any():  # Draw
        v = 0
        p = (p == 0).astype(int)
    else:  # Loss
        v = -1
        p[p < np.amax(p)] = 0
        p[p != 0] = 1
    p = p / sum(p)
    return v, p
