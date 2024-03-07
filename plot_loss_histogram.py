import numpy as np
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_value.value_loss import value_loss
from src.plotting.plot_utils import BarFigure, incremental_bin
from src.general.general_utils import models_path, game_path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
Value loss histogram
"""

# Choose game type:
game_num = 0
games = ['connect_four', 'pentago', 'oware', 'checkers']
env = games[game_num]
path = models_path()

par = np.load('src/config/parameter_counts/'+env+'.npy')
font = 18
font_num = 16

figure = BarFigure(par, 
                   x_label='State rank', 
                   y_label='Loss', 
                   title='Value loss on the train set', 
                   text_font=font, 
                   number_font=font_num)
figure.preamble()
color_nums = figure.colorbar_colors()

data_labels = [0, 1, 2, 3, 4, 5, 6] # for oware no 6
#data_labels = [6]
n_copies = 4

# initialize bins to cover a range definitely larger than what you'll need:
bins = incremental_bin(10**10)
widths = (bins[1:] - bins[:-1])
x = bins[:-1] + widths/2

state_counter = StateCounter(env, save_serial=True, save_value=True)

for label in data_labels:
    bin_counts = np.zeros(len(x))
    loss_sums = np.zeros(len(x))
    for copy in range(n_copies):
        model_name = f'q_{label}_{copy}'
        print(model_name)
        model_path = path + game_path(env) + model_name + '/'
        state_counter.reset_counters()
        state_counter.collect_data(path=model_path, max_file_num=20)
        state_counter.normalize_counters()

        state_counter.prune_low_frequencies(2)
        # consider pruning more, and checking that the max rank is more or less similar between all agents with the
        # same label. to avoid averaging different-frequency states together.

        loss = value_loss(env, model_path, state_counter=state_counter)

        ranks = np.arange(len(loss)) + 1
        # Calculate histogram.
        # np.histogram counts how many elements of 'ranks' fall in each bin.
        # by specifying 'weights=loss', you can make it sum losses instead of counting.
        bin_counts += np.histogram(ranks, bins=bins)[0]
        loss_sums += np.histogram(ranks, bins=bins, weights=loss)[0]
    # Divide sum to get average:
    mask = np.nonzero(bin_counts)
    loss_averages = loss_sums[mask] / bin_counts[mask]

    #plot_label = 'N = 10^{a:.2f}'.format(a=np.log10(par[label]))
    plt.scatter(x[mask], loss_averages,
                s=4, color=cm.viridis(color_nums[label]))

plt.xscale('log')
plt.yscale('log')
figure.epilogue()
figure.save('value_loss_scatter')

plt.yscale('linear')
figure.epilogue()   
figure.save('value_loss_scatter_semilog')

