import numpy as np
import pickle
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import Figure, incremental_bin
import matplotlib.pyplot as plt

"""
Plot turn related data: how late/early in the game do states appear.
"""

# Choose game type:
game_num = 2
games = ['connect_four', 'pentago', 'oware', 'checkers']

env = games[game_num]

plot_extra = False
save_data = False

data_labels = [0, 1, 2, 3, 4, 5, 6]  # for oware no 6
# data_labels = [0,1]

#state_counter = gather_data(env, data_labels, max_file_num=20, save_turn_num=True)
state_counter = gather_data(env, data_labels, mod='cutoff', max_file_num=20, save_turn_num=True)
state_counter.prune_low_frequencies(10)
turns_played = state_counter.turns_played
turns_to_end = state_counter.turns_to_end
board_counter = state_counter.frequencies

# Turns analysis:
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


print('Plotting turn distributions')
fig = Figure(x_label='State rank', text_font=font, number_font=font_num)

def plot_turns(y, name, y_label, y_logscale=False):
    fig.fig_num += 1
    fig.preamble()
    plt.scatter(x, y, s=40 * 3 / (10 + x), alpha=1, color='green')
    plt.xscale('log')
    if y_logscale:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    fig.y_label = y_label
    fig.epilogue()
    fig.save(name)


if plot_extra:
    print('plot turns taken so far')
    y = sort_by_frequency(data=turns_played, counter=board_counter)
    y = np.cumsum(y) / (np.arange(n) + 1)
    plot_turns(y, name='turns_taken', y_label='Average turn num.')

    print('plot turns until end of game')
    y = sort_by_frequency(data=turns_to_end, counter=board_counter)
    y = np.cumsum(y) / (np.arange(n) + 1)
    plot_turns(y, name='turns_left', y_label='Average turns until end')

print('Plot raw data (to see U-shape)')
y = sort_by_frequency(data=turns_played, counter=board_counter)
if save_data:
    with open('../plot_data/turns/raw_turns_' + env + '.pkl', 'wb') as f:
        pickle.dump(y, f)
plot_turns(y, name='turns_taken_raw', y_label='Turn num.', y_logscale=True)
y = sort_by_frequency(data=turns_to_end, counter=board_counter)
if save_data:
    with open('../plot_data/turns/raw_turns_to_end_' + env + '.pkl', 'wb') as f:
        pickle.dump(y, f)
plot_turns(y, name='turns_to_end_raw', y_label='Turns left', y_logscale=True)

# Plot percentage of late turns:
bins = incremental_bin(10 ** 10)
widths = (bins[1:] - bins[:-1])
bin_x = bins[:-1] + widths / 2

turn_mask = state_counter.late_turn_mask(threshold=40)
late_states = np.histogram(x[turn_mask], bins=bins)[0]
all_states = np.histogram(x, bins=bins)[0]

mask = np.nonzero(all_states)
ratio = late_states[mask] / all_states[mask]
bin_x = bin_x[mask]
if save_data:
    with open('../plot_data/turns/turn_ratio_' + env + '.pkl', 'wb') as f:
        pickle.dump([bin_x, ratio], f)

fig.fig_num += 1
fig.preamble()
plt.plot(bin_x, ratio)
plt.xscale('log')
plt.yscale('linear')
fig.y_label = 'Late turn ratio'
fig.x_label = 'State rank'
fig.epilogue()
fig.save('late_turn_ratio')
