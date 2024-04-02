import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import Figure, BarFigure, incremental_bin


"""
Plot turn related data: how late/early in the game do states appear.
"""
# theres a cool plot to be made here, overlaying data from different cutoff values.
# At 50, there is a tiny change, and by 80 the change is very visible, but still small.
# overlay the ratio plot, and also the turn num. plot.

# Choose game type:
game_num = 2
games = ['connect_four', 'pentago', 'oware', 'checkers']

env = games[game_num]

save_data = True

data_labels = [0, 1, 2, 3, 4, 5, 6]  # for oware no 6
# data_labels = [0,1]

cutoffs = [50, 65, 80, 100, 120, 140, 160, 180, 200]
freqs = {cutoff: [] for cutoff in cutoffs}
ys = {cutoff: [] for cutoff in cutoffs}
ratio_data = {cutoff: [] for cutoff in cutoffs}


def analyze_counts(cutoff):
    print('Analyzing cutoff:', cutoff)
    state_counter = gather_data(env, data_labels, cutoff=cutoff, max_file_num=10, save_turn_num=True)
    state_counter.prune_low_frequencies(10)
    turns_played = state_counter.turns_played
    board_counter = state_counter.frequencies

    freqs[cutoff] = np.array([item[1] for item in board_counter.most_common()])
    ys[cutoff] = sort_by_frequency(data=turns_played, counter=board_counter)

    # Plot percentage of late turns:
    bins = incremental_bin(10 ** 10)
    widths = (bins[1:] - bins[:-1])
    bin_x = bins[:-1] + widths / 2
    x = np.arange(len(board_counter)) + 1
    turn_mask = state_counter.late_turn_mask(threshold=40)
    late_states = np.histogram(x[turn_mask], bins=bins)[0]
    all_states = np.histogram(x, bins=bins)[0]

    mask = np.nonzero(all_states)
    ratio = late_states[mask] / all_states[mask]
    bin_x = bin_x[mask]
    ratio_data[cutoff] = (bin_x, ratio)


for cutoff in cutoffs:
    analyze_counts(cutoff)
if save_data:
    with open('../plot_data/turns/cutoff_ablation_' + env + '.pkl', 'wb') as f:
        pickle.dump([freqs, ys, ratio_data], f)
font = 18 - 2
font_num = 16 - 2

def colorbar():
    norm = matplotlib.colors.Normalize(vmin=min(cutoffs), vmax=max(cutoffs))
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=norm)
    cbar = plt.colorbar(sm)
    cbar.ax.tick_params(labelsize=font_num)
    cbar.ax.set_ylabel('Cutoff value', rotation=90, fontsize=font)
    return sm

print('Plotting zipf distribution')
fig = Figure(x_label='State rank',
             y_label='Frequency',
             text_font=font,
             number_font=font_num)
fig.preamble()
sm = colorbar()
for cutoff in cutoffs[::-1]:
    freq = freqs[cutoff]
    x = np.arange(len(freq)) + 1
    plt.scatter(x, freq, s=40 / (10 + x), color = sm.to_rgba(cutoff))  # color gradient
plt.xscale('log')
plt.yscale('log')
fig.epilogue()
fig.save('zipf_distribution_cutoff')

print('Plotting turn distributions')
fig = Figure(x_label='State rank', text_font=font, number_font=font_num)
fig.fig_num += 1
fig.preamble()
sm = colorbar()
for cutoff in cutoffs[::-1]:
    y = ys[cutoff]
    x = np.arange(len(y)) + 1
    plt.scatter(x, y, s=40 * 3 / (10 + x), color = sm.to_rgba(cutoff))
plt.xscale('log')
plt.yscale('log')
fig.y_label = 'Turn num.'
fig.epilogue()
fig.save('turns_taken_cutoff')

fig.fig_num += 1
fig.preamble()
sm = colorbar()
for cutoff in cutoffs[::-1]:
    bin_x, ratio = ratio_data[cutoff]
    plt.plot(bin_x, ratio, color = sm.to_rgba(cutoff))  # color gradient
plt.xscale('log')
plt.yscale('linear')
fig.y_label = 'Late turn ratio'
fig.x_label = 'State rank'
fig.epilogue()
fig.save('late_turn_ratio_cutoff')
