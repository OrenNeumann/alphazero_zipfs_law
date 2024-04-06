import numpy as np
import subprocess
from subprocess import PIPE
import time
import pickle
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.plotting.plot_utils import Figure, incremental_bin
from src.general.general_utils import models_path
import collections
import scipy
import matplotlib.pyplot as plt
#from tqdm import tqdm
import pyspiel
"""
Time how long it takes for alpha beta pruning to analyze connect4 states.
"""

game = pyspiel.load_game('connect_four')

def time_alpha_beta_pruning(states, max_time=2*60):
    """ Time the solver on states until max_time seconds have passed."""
    times = []
    beginning = time.time()
    for state in states:
        if time.time() - beginning > max_time:
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
    if len(times) == 1:
        gstd = 1
    else:
        gstd = scipy.stats.gstd(times)
    return np.mean(times), np.std(times), gstd
    

def save_pruning_time():
    # Gather states
    generate = True
    if generate:
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
    standard_devs = []
    gstds = []
    indices = []
    bin_range = np.arange(0,len(bins))#np.arange(1,200)#np.arange(20,50)
    rng = np.random.default_rng()
    #for i in tqdm(indices,desc='Calculating times'):
    for i in bin_range:
        print('loop',i)
        if int(bins[i+1]) > len(keys):
            break
        state_keys = keys[int(bins[i]):int(bins[i+1])]
        n = min(100,len(state_keys))
        print('n=',n)
        sample = rng.choice(state_keys, n, replace=False) 
        if len(sample) < 10:
            sample = np.concatenate([sample,sample])
        states = [game.deserialize_state(serial_states[key]) for key in sample]
        if i < 15:
            t, stdv, gstd= time_alpha_beta_pruning(states, max_time=3*60)
        else:
            t, stdv, gstd= time_alpha_beta_pruning(states)
        times.append(t)
        standard_devs.append(stdv)
        gstds.append(gstd)
        indices.append(i)
    with open('../plot_data/ab_pruning/data.pkl', 'wb') as f:
        pickle.dump({'x': bin_x[indices],
                     'times': times,
                     'std': standard_devs,
                     'gstd': gstds}, f)
        
    fig = Figure(text_font=16,
                number_font=14,
                fig_num=3)
    fig.preamble()
    # Plot with geometric standard deviation error bars
    err = [np.array(times)*np.array(gstds), np.array(times)/np.array(gstds)]
    plt.errorbar(bin_x[indices], times, yerr=err, fmt='-o')
    plt.xscale('log')
    plt.yscale('log')
    fig.y_label = 'CPU time (s)'
    fig.x_label = 'State rank'
    fig.title = 'Alpha-beta pruning resources'
    fig.epilogue()
    fig.save('search_time')
    

def generate_states():
    state_counter = StateCounter(env='connect_four', save_serial=True)
    for label in [0, 1, 2, 3, 4, 5, 6]:
        num = str(label)
        path = models_path() + '/connect_four_10000/q_' + num + '_0/'
        state_counter.collect_data(path=path, max_file_num=5)

    # Prune
    state_counter.prune_low_frequencies(10)
    with open('../plot_data/ab_pruning/counter.pkl', 'wb') as f:
        pickle.dump(state_counter, f)
    font = 16
    font_num = 14
    print('Plotting zipf distribution')
    fig = Figure(x_label='State rank',
                y_label='Frequency',
                text_font=font,
                number_font=font_num,
                legend=True,
                fig_num=2)
    freq = np.array([item[1] for item in state_counter.frequencies.most_common()])
    x = np.arange(len(state_counter.frequencies)) + 1
    plt.scatter(x, freq, s=40 / (10 + x))
    plt.xscale('log')
    plt.yscale('log')
    fig.epilogue()
    fig.save('zipf_distribution')
