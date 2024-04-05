import numpy as np
import subprocess
from subprocess import PIPE
import time
import random
import pickle
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.plotting.plot_utils import Figure, incremental_bin
from src.general.general_utils import models_path, game_path
import collections

import matplotlib.pyplot as plt
from tqdm import tqdm
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
        _ = subprocess.run(["src/alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE)
        time_end = time.time()
        times.append(time_end - time_start)
    return np.mean(times)
    

def save_pruning_time():
    env = 'connect_four'
    board_counter = collections.Counter()
    serial_states = dict()
    state_counter = StateCounter(env, save_serial=True)
    for label in [0, 2, 4, 6]:
        num = str(label)
        path = models_path() + '/connect_four_10000/q_' + num + '_0/'
        state_counter.collect_data(path=path, max_file_num=1)
        # add counts to the counter, and update new serial states:
        board_counter.update(state_counter.frequencies)
        serial_states.update(state_counter.serials)

    n = len(board_counter)
    x = np.arange(n) + 1

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
    indices = np.arange(4,6)
    for i in tqdm(indices,desc='Calculating times'):
        states = keys[int(bin_x[i]):int(bin_x[i+1])]
        states = random.sample(states,k=min(10,len(states)))
        times.append(time_alpha_beta_pruning(states))

    fig.fig_num += 1
    fig.preamble()
    plt.plot(bin_x[indices], times)
    plt.xscale('log')
    plt.yscale('log')
    fig.y_label = 'CPU time (s)'
    fig.x_label = 'State rank'
    fig.epilogue()
    fig.save('search_time')
    
