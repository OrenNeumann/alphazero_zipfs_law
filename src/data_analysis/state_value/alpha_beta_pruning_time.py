import numpy as np
import subprocess
from subprocess import PIPE
import time
import pickle
from src.data_analysis.gather_agent_data import gather_data
from src.plotting.plot_utils import incremental_bin
from scipy.stats import gmean, gstd
import pyspiel
import os

"""
Time how long it takes for alpha beta pruning to analyze connect4 states.
This code requires that the connect four opening book ("7x6.book") is NOT present in the parent directory.
"""

game = pyspiel.load_game('connect_four')

def time_alpha_beta_pruning(states, max_time: int = 2*60) -> tuple[float, float, float, float]:
    """ Time the solver on states until max_time seconds have passed."""
    times = []
    beginning = time.time()
    for state in states:
        if time.time() - beginning > max_time:
            if len(times) != 1:
                break
        moves = ""
        for action in state.full_history():
            # The solver starts counting moves from 1:
            moves += str(action.action + 1) 
        time_start = time.time()
        _ = subprocess.run(["src/alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE)
        time_end = time.time()
        times.append(time_end - time_start)
    print(times)
    #gm_stdev = 1 if len(times)==1 else gstd(times)
    return np.mean(times), np.std(times), gmean(times), gstd(times)
    

def save_pruning_time(generate_counter=False) -> None:
    print("""Generating alpha-beta pruning data.
          NOTICE: Generating this data from scratch takes a prohibitively long amount of time,
          since it measures the wall-time of exponentially-longer ab-pruning runs.""")
    if os.path.isfile('./7x6.book'):
        raise ValueError("The connect four opening book ('7x6.book') should not be present in the parent directory.")
    # Gather states
    if generate_counter:
        generate_states()
    with open('../plot_data/ab_pruning/counter.pkl', 'rb') as f:
        state_counter = pickle.load(f)
    board_counter = state_counter.frequencies
    serial_states = state_counter.serials

    # Calculate solver times
    bins = incremental_bin(10 ** 10)
    widths = (bins[1:] - bins[:-1])
    bin_x = bins[:-1] + widths / 2
    keys = np.array([key for key,_ in board_counter.most_common()])
    times = []
    g_mean_times = []
    standard_devs = []
    gstds = []
    indices = []
    bin_range = np.arange(0,len(bins))
    rng = np.random.default_rng()
    for i in bin_range:
        print('loop',i)
        if int(bins[i+1]) > len(keys):
            break
        state_keys = keys[int(bins[i]):int(bins[i+1])]
        n = min(100,len(state_keys))
        print('n=',n)
        sample = rng.choice(state_keys, n, replace=False) 
        if len(sample) == 1:
            sample = np.concatenate([sample,sample])
        states = [game.deserialize_state(serial_states[key]) for key in sample]
        if i < 5:
            t, stdv, g_t, gstd= time_alpha_beta_pruning(states, max_time=4*60)
        elif i < 15:
            t, stdv, g_t, gstd= time_alpha_beta_pruning(states, max_time=3*60)
        else:
            t, stdv, g_t, gstd= time_alpha_beta_pruning(states)
        times.append(t)
        standard_devs.append(stdv)
        gstds.append(gstd)
        g_mean_times.append(g_t)
        indices.append(i)
        if i%10 == 0:
            with open('../plot_data/ab_pruning/data.pkl', 'wb') as f:
                pickle.dump({'x': bin_x[indices],
                            'times': times,
                            'g_mean': g_mean_times,
                            'std': standard_devs,
                            'gstd': gstds}, f)
        
    with open('../plot_data/ab_pruning/data.pkl', 'wb') as f:
        pickle.dump({'x': bin_x[indices],
                     'times': times,
                     'g_mean': g_mean_times,
                     'std': standard_devs,
                     'gstd': gstds}, f)
    

def generate_states() -> None:
    state_counter = gather_data(env='connect_four', labels=[0, 1, 2, 3, 4, 5, 6], max_file_num=10, save_serial=True)
    state_counter.prune_low_frequencies(10)
    with open('../plot_data/ab_pruning/counter.pkl', 'wb') as f:
        pickle.dump(state_counter, f)

