from collections import Counter
import re
import numpy as np
import pickle
from src.data_analysis.gather_agent_data import gather_data
from src.plotting.plot_utils import aligned_title
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots

""" Plotting frequencies of capture differences (=diff in number of pieces captured by each player).
plots the capture frequencies for both pruned and unpruned in the same plot. shows turns is not the only 
frequency phenomenon."""

plt.style.use(['science','nature'])

def plot_capture_differences(load_data=True,res=300):
    print('~~~~~~~~~~~~~~~~~~~ Plotting capture difference (appendix) ~~~~~~~~~~~~~~~~~~~')
    tf =12
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    env = 'oware'

    if not load_data:
        _generate_states(env)

    ax = axs[0]

    with open(f'../plot_data/capture_diff/{env}_counter.pkl', 'rb') as f:
        state_counter = pickle.load(f)

    diff_counter = _get_diff_counter(state_counter, env)
    x, y = _get_curve(diff_counter)
    ax.plot(x, y,color='dodgerblue', label='All states')

    print('Plotting pruned capture difference:')
    state_counter.prune_low_frequencies(10)
    diff_counter = _get_diff_counter(state_counter, env)
    x, y = _get_curve(diff_counter)
    ax.plot(x, y, color='gold', label='Only high-frequency states')

    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Capture difference',fontsize=tf)
    ax.set_ylabel('Frequency',fontsize=tf)
    ax.tick_params(axis='both', which='major', labelsize=tf-2)
    aligned_title(axs[0], r"$\bf{A.}$ Oware", tf+4)
    
    fig.tight_layout()
    fig.savefig('./plots/capture_difference.png', dpi=res)

def _get_curve(diff_counter):
    x = list(diff_counter.keys())
    y = list(diff_counter.values())
    # sort by x
    x, y = zip(*sorted(zip(x, y)))
    return x, y
    x=[]
    y=[]
    for key, count in diff_counter.most_common():
        x.append(key)
        y.append(count)
    return x, y

def _get_diff_counter(counter, env):
    board_counter = counter.frequencies
    keys = np.array([key for key,_ in board_counter.most_common()])
    diffs = np.zeros(len(keys))
    diff_counter = Counter()
    for i, key in enumerate(tqdm(keys, desc=f'Calc. {env} capture differences')):
        diffs[i], _ = _capture_diff(key,env)
        diff_counter[diffs[i]] += board_counter[key]
    return diff_counter

def _capture_diff(state_str, env):
    """ Returns (capture difference, total captured) for a state string."""
    if env == 'oware':
        score_x = int(state_str.split('\n')[0][17:19])
        score_o = int(state_str.split('\n')[-2][17:19])
    else:
        raise NameError('Environment '+ env + 'not supported')
    return np.abs(score_x - score_o), score_x+score_o


def _generate_states(env):
    state_counter = gather_data(env, labels=[0, 1, 2, 3, 4, 5, 6], max_file_num=10)
    #state_counter.prune_low_frequencies(10)
    with open(f'../plot_data/capture_diff/{env}_counter.pkl', 'wb') as f:
        pickle.dump(state_counter, f)