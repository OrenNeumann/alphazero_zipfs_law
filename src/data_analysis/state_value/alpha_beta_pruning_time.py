import numpy as np
import subprocess
from subprocess import PIPE
import time
import pickle
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.plotting.plot_utils import Figure, incremental_bin
from src.general.general_utils import models_path
import collections

import matplotlib.pyplot as plt
from tqdm import tqdm
import pyspiel
"""
Time how long it takes for alpha beta pruning to analyze connect4 states.
"""

game = pyspiel.load_game('connect_four')

def time_alpha_beta_pruning(states):
    """ Time the solver on states until 2 minutes have passed."""
    times = []
    total_time = time.time()
    for state in states:
        if time.time() - total_time > 2*60:
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
    return np.mean(times), np.std(times)
    

def save_pruning_time():
    env = 'connect_four'
    board_counter = collections.Counter()
    serial_states = dict()
    state_counter = StateCounter(env, save_serial=True)
    for label in [0, 2, 4, 6]:
        num = str(label)
        path = models_path() + '/connect_four_10000/q_' + num + '_0/'
        state_counter.collect_data(path=path, max_file_num=2)
        # add counts to the counter, and update new serial states:
        board_counter.update(state_counter.frequencies)
        serial_states.update(state_counter.serials)

    state_counter.prune_low_frequencies(10)

    font = 18 - 2
    font_num = 16 - 2

    print('Plotting zipf distribution')
    fig = Figure(x_label='State rank',
                y_label='Frequency',
                text_font=font,
                number_font=font_num,
                legend=True,
                fig_num=2)
    
    freq = np.array([item[1] for item in board_counter.most_common()])
    x = np.arange(len(board_counter)) + 1
    plt.scatter(x, freq, s=40 / (10 + x))
    plt.xscale('log')
    plt.yscale('log')
    fig.epilogue()
    fig.save('zipf_distribution')

    bins = incremental_bin(10 ** 10)
    widths = (bins[1:] - bins[:-1])
    bin_x = bins[:-1] + widths / 2
    keys = np.array([key for key,_ in board_counter.most_common()])
    times = []
    standard_devs = []
    indices = np.arange(20,50)
    #for i in tqdm(indices,desc='Calculating times'):
    for i in indices:
        print('loop',i)
        state_keys = keys[int(bin_x[i]):int(bin_x[i+1])]
        n = min(100,len(state_keys))
        print('n=',n)
        sample = np.random.choice(state_keys.shape[0], n, replace=False)  
        states = [game.deserialize_state(serial_states[key]) for key in state_keys[sample]]
        t, stdv = time_alpha_beta_pruning(states)
        times.append(t)
        standard_devs.append(stdv)
    with open('../plot_data/ab_pruning.pkl', 'wb') as f:
        pickle.dump({'x': bin_x[indices],
                     'times': times,
                     'std': standard_devs,
                     'counter': state_counter}, f)
    fig.fig_num += 1
    fig.preamble()
    #plt.plot(bin_x[indices], times)
    plt.errorbar(bin_x[indices], times, yerr=[standard_devs, standard_devs], fmt='-o')
    plt.xscale('log')
    plt.yscale('log')
    fig.y_label = 'CPU time (s)'
    fig.x_label = 'State rank'
    fig.epilogue()
    fig.save('search_time')
    
