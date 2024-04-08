from collections import Counter
import numpy as np
import pickle
from src.plotting.plot_utils import Figure
from src.data_analysis.gather_agent_data import gather_data
import matplotlib.pyplot as plt
from tqdm import tqdm

""" Plotting frequencies of capture differences (=diff in number of pieces captured by each player)."""


def plot_capture_differences():
    env = 'oware'
    # Gather states
    generate = False
    if generate:
        generate_states()
    with open('../plot_data/capture_diff/counter.pkl', 'rb') as f:
        state_counter = pickle.load(f)
    board_counter = state_counter.frequencies
    keys = np.array([key for key,_ in board_counter.most_common()])
    diffs = np.zeros(len(keys))
    captured = np.zeros(len(keys))
    diff_counter = Counter()
    for i, key in tqdm(enumerate(keys), desc='Calc. capture differences'):
        d = capture_diff(key,env)
        diffs[i], captured[i] = capture_diff(key,env)
        diff_counter[d] += board_counter[key]
    x = np.arange(len(keys)) + 1
    print('Plotting capture distribution')
    fig = Figure(x_label='State rank', y_label='Capture difference', text_font=16, number_font=14, fig_num=2)
    fig.preamble()
    plt.scatter(x, diffs, s=40 * 3 / (10 + x))
    plt.scatter(x, captured, s=40 * 3 / (10 + x))
    plt.xscale('log')
    plt.yscale('linear')
    fig.epilogue()
    fig.save('capture_diff_distribution')

    print('Plotting capture frequency')
    fig = Figure(x_label='Capture difference', y_label='Frequency', text_font=16, number_font=14, fig_num=3)
    fig.preamble()
    xd=[]
    y=[]
    for key, count in diff_counter.most_common():
        xd.append(key)
        y.append(count)
    plt.scatter(xd, y)
    plt.xscale('linear')
    plt.yscale('log')
    fig.epilogue()
    fig.save('capture_diff_frequency')


def capture_diff(state_str, env):
    # calc capture difference in abs value
    if env == 'oware':
        score_x = int(state_str.split('\n')[0][17:19])
        score_o = int(state_str.split('\n')[0][17:19])
    else:
        raise NameError('Environment '+ env + 'not supported')
    return np.abs(score_x - score_o), score_x+score_o



def generate_states(env='oware'):
    state_counter = gather_data(env, labels=[0, 1, 2, 3, 4, 5, 6], max_file_num=10)
    state_counter.prune_low_frequencies(10)
    with open('../plot_data/capture_diff/counter.pkl', 'wb') as f:
        pickle.dump(state_counter, f)
    font = 16
    font_num = 14
    print('Plotting zipf distribution')
    fig = Figure(x_label='State rank',
                y_label='Frequency',
                text_font=font,
                number_font=font_num,
                legend=True,
                fig_num=1)
    freq = np.array([item[1] for item in state_counter.frequencies.most_common()])
    x = np.arange(len(state_counter.frequencies)) + 1
    plt.scatter(x, freq, s=40 / (10 + x))
    plt.xscale('log')
    plt.yscale('log')
    fig.epilogue()
    fig.save('zipf_distribution')

plot_capture_differences()
