import numpy as np
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_value.value_loss import value_loss
from src.plotting.plot_utils import BarFigure, incremental_bin
from src.general.general_utils import models_path, game_path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
Value loss histogram only on late-game states
"""

# Choose game type:
game_num = 2
games = ['connect_four', 'pentago', 'oware', 'checkers']
env = games[game_num]
path = models_path()

par = np.load('src/config/parameter_counts/'+env+'.npy')
font = 18
font_num = 16

figure = BarFigure(par, 
                   x_label='State rank', 
                   y_label='Loss', 
                   title='Value loss on late-game states', 
                   text_font=font, 
                   number_font=font_num)
figure.preamble()
color_nums = figure.colorbar_colors()

data_labels = [0, 1, 2, 3, 4, 5] # for oware no 6 (get from cluster)
#data_labels = [0]
n_copies = 1#6

# initialize bins to cover a range definitely larger than what you'll need:
bins = incremental_bin(10**10)
widths = (bins[1:] - bins[:-1])
x = bins[:-1] + widths/2

state_counter = StateCounter(env, save_serial=True, save_value=True, save_turn_num=True)
total_loss = np.zeros([len(data_labels),n_copies])
total_counts = np.zeros([len(data_labels),n_copies])

for label in data_labels:
    bin_counts = np.zeros(len(x))
    loss_sums = np.zeros(len(x))
    for copy in range(n_copies):
        model_name = f'q_{label}_{copy}'
        print(model_name)
        model_path = path + game_path(env) + model_name + '/'
        state_counter.reset_counters()
        state_counter.collect_data(path=model_path, max_file_num=39)
        state_counter.normalize_counters()

        state_counter.prune_low_frequencies(threshold=10)
        #state_counter.prune_early_turns(threshold=40)
        turn_mask = state_counter.late_turn_mask(threshold=40)

        #freq = np.array([c for k, c in state_counter.frequencies.most_common()])

        loss = value_loss(env, model_path, state_counter=state_counter)

        ranks = np.arange(len(loss)) + 1
        # Calculate histogram.
        # np.histogram counts how many elements of 'ranks' fall in each bin.
        # by specifying 'weights=loss', you can make it sum losses instead of counting.
        bin_counts += np.histogram(ranks[turn_mask], bins=bins)[0]
        loss_sums += np.histogram(ranks, bins=bins, weights=loss*turn_mask)[0]
        #loss_sums += np.histogram(ranks, bins=bins, weights=loss*freq)[0]
    # Divide sum to get average:
    mask = np.nonzero(bin_counts)
    loss_averages = loss_sums[mask] / bin_counts[mask]

    # Line plot:
    plt.plot(x[mask], loss_averages,
                color=cm.viridis(color_nums[label]))
    

plt.xscale('log')
plt.yscale('log')
figure.epilogue()
figure.save('lategame_loss_scatter')

plt.yscale('linear')
figure.epilogue()   
figure.save('lategame_scatter_semilog')
