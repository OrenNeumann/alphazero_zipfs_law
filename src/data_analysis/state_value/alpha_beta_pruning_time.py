import numpy as np
import subprocess
from subprocess import PIPE
import time

"""
Time how long it takes for alpha beta pruning to analyze connect4 states.
"""

def time_alpha_beta_pruning(states):
    times = []
    for state in states:
        moves = ""
        for action in state.full_history():
            # The solver starts counting moves from 1:
            moves += str(action.action + 1) 
        time_start = time.time()
        _ = subprocess.run(["alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE)
        time_end = time.time()
        times.append(time_end - time_start)
    return np.mean(times)
    