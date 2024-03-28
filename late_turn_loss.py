import numpy as np
import pickle
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_value.value_loss import value_loss
from src.plotting.plot_utils import BarFigure, incremental_bin
from src.general.general_utils import models_path, game_path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
Value loss histogram only on late-game states
"""

load_data = False

# Choose game type:
game_num = 3
games = ['connect_four', 'pentago', 'oware', 'checkers']
env = games[game_num]
path = models_path()


loss_types = ('later_turns','early_turns','every_state')
data_labels = [0, 1, 2, 3, 4, 5, 6] # for oware no 6 (get from cluster)
#data_labels = [0,2,5]
n_copies = 1#6

# initialize bins to cover a range definitely larger than what you'll need:
bins = incremental_bin(10**10)
widths = (bins[1:] - bins[:-1])
x = bins[:-1] + widths/2

def calc_loss_curves():
    state_counter = StateCounter(env, save_serial=True, save_value=True, save_turn_num=True)
    loss_values = {label: {k: None for k in loss_types} for label in data_labels}
    rank_values = {label: {k: None for k in loss_types} for label in data_labels}
    for label in data_labels:
        bin_count = {k: np.zeros(len(x)) for k in loss_types}
        loss_sums = {k: np.zeros(len(x)) for k in loss_types}
        for copy in range(n_copies):
            model_name = f'q_{label}_{copy}'
            print(model_name)
            model_path = path + game_path(env) + model_name + '/'
            state_counter.reset_counters()
            state_counter.collect_data(path=model_path, max_file_num=70)
            state_counter.normalize_counters()

            state_counter.prune_low_frequencies(threshold=1)#10
            turn_mask = state_counter.late_turn_mask(threshold=40)

            loss = value_loss(env, model_path, state_counter=state_counter)

            ranks = np.arange(len(loss)) + 1

            # Calculate histogram.
            # np.histogram counts how many elements of 'ranks' fall in each bin.
            bin_count['later_turns'] += np.histogram(ranks[turn_mask], bins=bins)[0]
            loss_sums['later_turns'] += np.histogram(ranks, bins=bins, weights=loss*turn_mask)[0]
            bin_count['early_turns'] += np.histogram(ranks[~turn_mask], bins=bins)[0]
            loss_sums['early_turns'] += np.histogram(ranks, bins=bins, weights=loss*(~turn_mask))[0]
            bin_count['every_state'] += np.histogram(ranks, bins=bins)[0]
            loss_sums['every_state'] += np.histogram(ranks, bins=bins, weights=loss)[0]

        # Divide sum to get average:
        for t in loss_types:
            mask = np.nonzero(bin_count[t])
            loss_values[label][t] = loss_sums[t][mask] / bin_count[t][mask]
            rank_values[label][t] = x[mask]
        
    with open('../plot_data/value_loss/loss_curves_'+env+'.pkl', 'wb') as f:
        pickle.dump([loss_values,rank_values], f)

    return loss_values, rank_values

def smooth(vec):
    """return a smoothed vec with values averaged with their neighbors."""
    a = 0.5
    new_vec = np.zeros(len(vec))
    new_vec[0] = (vec[0] + a*vec[1])/ (1 + a)
    for i in range(len(vec)-2):
        new_vec[i+1] = (a*vec[i] + vec[i+1] + a*vec[i+2]) / (1 + 2*a)
    new_vec[-1] = (vec[-1] + a*vec[-2])/ (1 + a)
    return new_vec

if load_data:
    print('Loading')
    with open('../plot_data/value_loss/loss_curves_'+env+'.pkl', "rb") as f:
        loss_values, rank_values =  pickle.load(f)
else:
    loss_values, rank_values = calc_loss_curves()

print('Plotting...')

par = np.load('src/config/parameter_counts/'+env+'.npy')
font = 18
font_num = 16

figure = BarFigure(par, 
                   x_label='State rank', 
                   y_label='Loss', 
                   #title='Value loss on late-game states', 
                   text_font=font, 
                   number_font=font_num)

color_nums = figure.colorbar_colors()

for t in loss_types:
    figure.title = 'Value loss '+ t
    figure.preamble()
    for label in data_labels:
        x = rank_values[label][t]
        y = loss_values[label][t]
        y = smooth(y)
        plt.plot(x, y, color=cm.viridis(color_nums[label]))

    plt.xscale('log')
    plt.yscale('log')
    figure.epilogue()
    figure.save('value_loss_'+t)
    figure.fig_num += 1