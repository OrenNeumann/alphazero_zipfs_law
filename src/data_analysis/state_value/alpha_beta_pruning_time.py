import numpy as np
import subprocess
from subprocess import PIPE
import time
import pickle
from src.plotting.plot_utils import Figure, incremental_bin
from src.data_analysis.gather_agent_data import gather_data
from scipy.stats import gmean, gstd
import matplotlib.pyplot as plt
#from tqdm import tqdm
import pyspiel

"""
Time how long it takes for alpha beta pruning to analyze connect4 states.
This code requires that the connect four opening book ("7x6.book") is NOT present in the parent directory.
"""

game = pyspiel.load_game('connect_four')

def time_alpha_beta_pruning(states, max_time=2*60):
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
    

def save_pruning_time():
    # Gather states
    generate = False
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
    g_mean_times = []
    standard_devs = []
    gstds = []
    indices = []
    bin_range = np.arange(0,len(bins))
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
        
    fig = Figure(text_font=16,
                number_font=14,
                fig_num=3)
    fig.preamble()
    # Plot with geometric standard deviation error bars
    err = [np.array(g_mean_times)*np.array(gstds), np.array(g_mean_times)/np.array(gstds)]
    plt.errorbar(bin_x[indices], g_mean_times, yerr=err, fmt='-o')
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
    state_counter = gather_data(env='connect_four', labels=[0, 1, 2, 3, 4, 5, 6], max_file_num=10, save_serial=True)
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
