import numpy as np
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.game_data_analysis import process_games
from src.data_analysis.value_prediction import get_model_value_estimator
from src.plotting.plot_utils import figure_preamble, figure_epilogue, BarFigure
from src.general.general_utils import incremental_bin, models_path, game_path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from collections import Counter

"""
Value loss histogram
"""

# Choose game type:
game_num = 0
games = ['connect4', 'pentago', 'oware', 'checkers']
env = games[game_num]
path = models_path()
num_chunks = 20


def value_loss(path_model, board_count, info):
    """
    Calculate the value loss of a model on all states, sorted by rank.
    """
    serials = info['serials']
    real_values = info['values']
    model_values = get_model_value_estimator(env, path_model)
    sorted_serials = []
    z = []
    for key, _ in board_count.most_common():
        sorted_serials.append(serials[key])
        z.append(real_values[key])
    z = np.array(z)

    # Chunk data to smaller pieces to save memory:
    chunk_size = len(sorted_serials) // num_chunks
    data_chunks = [sorted_serials[i:i + chunk_size] for i in range(0, len(sorted_serials), chunk_size)]
    vl = []
    for chunk in tqdm(data_chunks, desc='Estimating model loss'):
        vl.append(model_values(chunk))
    v = np.concatenate(vl)

    return (z - v) ** 2


par = np.load('src/config/parameter_counts/'+env+'.npy')
font = 18
font_num = 16

figure = BarFigure(par, 
                   x_label='State rank', 
                   y_label='Loss', 
                   title='Value loss on the train set', 
                   text_font=font, 
                   number_font=font_num, 
                   legend=True)
figure.figure_preamble()
color_nums = figure.colorbar_colors()
print(color_nums)

#data_labels = [0, 1, 2, 3, 4, 5, 6] # for oware no 6
data_labels = [6]
n_copies = 3

# initialize bins to cover a range definitely larger than what you'll need:
bins = incremental_bin(10**10)
widths = (bins[1:] - bins[:-1])
x =  bins[:-1] + widths/2

for label in data_labels:
    counts = np.zeros(len(x))
    loss_sums = np.zeros(len(x))
    for copy in range(n_copies):
        model_name = f'q_{label}_{copy}'
        print(model_name)
        model_path = path + game_path(env) + model_name + '/'
        board_counter, info = process_games(env, model_path, max_file_num=2, save_serial=True, save_value=True)

        # seems pruning 1's reduces by one OOM, 2's and 3's together by another OOM.
        #print('Counter length before pruning:', len(board_counter))
        board_counter = Counter({k: c for k, c in board_counter.items() if c >= 2})
        #print('Counter length after pruning: ', len(board_counter))

        loss = value_loss(model_path,board_count=board_counter, info=info)

        ranks = np.arange(len(loss)) + 1
        # Calculate histogram.
        # np.histogram counts how many elements of 'ranks' fall in each bin.
        # by specifying 'weights=loss', you can make it sum losses instead of counting.
        counts += np.histogram(ranks, bins=bins)[0]
        loss_sums += np.histogram(ranks, bins=bins, weights=loss)[0]
    # Divide sum to get average:
    mask = np.nonzero(counts)
    print(mask)
    loss_averages = loss_sums[mask] / counts[mask]
    print(loss_averages)
    print(x[mask])

    plt.scatter(x[mask], loss_averages,s=10,label=label, color=cm.viridis(color_nums[label]))

plt.xscale('log')
plt.yscale('log')
figure.figure_epilogue()

name = 'value_loss_scatter'
plt.savefig('plots/'+name+'.png', dpi=900)

"""
figure_preamble()
plt.bar(bins[:-1], loss_averages, widths)
plt.xscale('log')
plt.yscale('log')
figure_epilogue(x_label='State rank',
                y_label='Loss',
                title='Value loss of fully-trained agent',
                label_font=font,
                number_font=font_num)
name = 'value_loss_barplot'
plt.savefig('plots/'+name+'.png', dpi=900)
"""
