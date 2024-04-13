import pickle
import numpy as np
from scipy.datasets import face
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_value.value_loss import value_loss
from src.plotting.plot_utils import BarFigure, incremental_bin, gaussian_average
from src.general.general_utils import models_path, game_path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
Value loss histogram
"""

load = False

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
color_nums = figure.colorbar_colors()
figure.preamble()

data_labels = [0, 1, 2, 3, 4, 5, 6] # for oware no 6
#data_labels = [0]
n_copies = 6

if load:
    with open('../plot_data/value_loss/training_loss/loss_curves_'+env+'.pkl', 'rb') as f:
        loss_curves = pickle.load(f)
else:
    # initialize bins to cover a range definitely larger than what you'll need:
    bins = incremental_bin(10**10)
    widths = (bins[1:] - bins[:-1])
    x = bins[:-1] + widths/2


    state_counter = StateCounter(env, save_serial=True, save_value=True)
    total_loss = np.zeros([len(data_labels),n_copies])
    total_counts = np.zeros([len(data_labels),n_copies])
    loss_curves = dict()
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

            state_counter.prune_low_frequencies(10)#4
            # consider pruning more, and checking that the max rank is more or less similar between all agents with the
            # same label. to avoid averaging different-frequency states together.

            #freq = sorted(state_counter.frequencies.items(), key=lambda x: x[1], reverse=True)
            #freq = np.array([item[1] for item in freq])
            freq = np.array([item[1] for item in state_counter.frequencies.most_common()])

            loss = value_loss(env, model_path, state_counter=state_counter)
            loss_curves[model_name] = loss
            total_loss[label, copy] = np.sum(loss * freq)
            total_counts[label, copy] = np.sum(freq)

            ranks = np.arange(len(loss)) + 1
            # Calculate histogram.
            # np.histogram counts how many elements of 'ranks' fall in each bin.
            # by specifying 'weights=loss', you can make it sum losses instead of counting.
            bin_counts += np.histogram(ranks, bins=bins)[0]
            loss_sums += np.histogram(ranks, bins=bins, weights=loss)[0]
            #loss_sums += np.histogram(ranks, bins=bins, weights=loss*freq)[0]
        # Divide sum to get average:
        mask = np.nonzero(bin_counts)
        loss_averages = loss_sums[mask] / bin_counts[mask]
        plt.plot(x[mask], loss_averages,
                    color=cm.viridis(color_nums[label]))
    with open('../plot_data/value_loss/training_loss/loss_curves_'+env+'.pkl', 'wb') as f:
        pickle.dump(loss_curves, f)


print('Total loss L:', total_loss)
print('Total counts:', total_counts)
for copy in range(n_copies):
    total_loss[:, copy] /= total_counts[:, copy]
print('Average loss L:', total_loss)
    

plt.xscale('log')
plt.yscale('log')
figure.epilogue()
figure.save('value_loss_scatter')

plt.yscale('linear')
figure.epilogue()   
figure.save('value_loss_scatter_semilog')

figure.fig_num += 1
figure.preamble()
for label in data_labels:
    curves = [np.array(loss_curves[f'q_{label}_{copy}']) for copy in range(n_copies)]
    l = min([len(curve) for curve in curves])
    curves = [curve[:l] for curve in curves]
    y = gaussian_average(np.mean(curves, axis=0))
    plt.plot(np.arange(len(y))+1, y, color=cm.viridis(color_nums[label]))
plt.xscale('log')
plt.yscale('log')
figure.epilogue()
figure.save('value_loss_gaussian')

figure.fig_num += 1
figure.preamble()
for label in data_labels:
    curves = [np.array(loss_curves[f'q_{label}_{copy}']) for copy in range(n_copies)]
    l = min([len(curve) for curve in curves])
    curves = [curve[:l] for curve in curves]
    y = gaussian_average(np.mean(curves, axis=0))
    var = gaussian_average(np.var(curves, axis=0))
    plt.plot(np.arange(len(y))+1, y, color=cm.viridis(color_nums[label]))
    plt.fill_between(np.arange(len(y)) + 1, y - np.sqrt(var), y + np.sqrt(var), 
                     alpha=0.2, facecolor=cm.viridis(color_nums[label]))

plt.xscale('log')
plt.yscale('log')
figure.epilogue()
figure.save('value_loss_gaussian_error')