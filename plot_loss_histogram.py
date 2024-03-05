import numpy as np
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.utils import get_model_value_estimator, models_path
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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

#data_labels = [0, 1, 2, 3, 4, 5, 6] # for oware no 6
data_labels = [6]
board_counter, info = gather_data(env, data_labels, max_file_num=10, save_serial=True, save_value=True)
serials = info['serials']
real_values = info['values']


# seems pruning 1's reduces by one OOM, 2's and 3's together by another OOM.
print('Counter length before pruning:', len(board_counter))
board_counter = Counter({k: c for k, c in board_counter.items() if c >= 2})
print('Counter length after pruning: ', len(board_counter))

num_processes = 20
def multiprocess_values(data):
    return model_values(data)

# Value loss analysis:
print('loss part...')
#estimators = [0, 1, 2, 3, 4, 5, 6]
estimators = [6]
if len(estimators) != 1:
    raise Exception('only single agent ATM')
model_losses = dict()
for agent in estimators:
    path_model = path + 'connect_four_10000/q_' + str(agent) + '_0/'
    model_values = get_model_value_estimator(env, path_model)
    temp_losses = dict()

    temp_serials = []
    z = []
    for key, count in board_counter.most_common():
        temp_serials.append(serials[key])
        z.append(real_values[key])
    z = np.array(z)
    chunk_size = len(temp_serials) // num_processes
    data_chunks = [temp_serials[i:i + chunk_size] for i in range(0, len(temp_serials), chunk_size)]
    vl = []
    for chunk in tqdm(data_chunks, desc='Estimating model ' + str(agent) + ' loss'):
        vl.append(model_values(chunk))
    v = np.concatenate(vl)
    # Create a multiprocessing Pool
    #pool = Pool(processes=num_processes)
    #v = pool.map(multiprocess_values, data_chunks)

    model_losses[agent] = (z - v) ** 2
    """
    for key in tqdm(board_counter.keys(), desc='Estimating model' + str(agent) + ' loss'):
        v = model_value(serials[key])
        z = real_values[key]
        temp_losses[key] = (z - v) ** 2
    model_losses[agent] = temp_losses
    """

par = np.load('config/parameter_counts/'+env+'.npy')
font = 18
font_num = 16

def bar_figure_preamble(par,label_font=18, number_font=16):
    # colorbar plot cargo-cult code
    w, h = plt.figaspect(0.6)
    plt.figure(2, figsize=(w, h))
    plt.style.use(['grid'])
    plt.clf()
    #ax = plt.gca()
    #norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
    # create a scalarmappable from the colormap
    #sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    #cbar = plt.colorbar(sm)
    #cbar.ax.tick_params(labelsize=number_font)
    #cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=label_font)


def incremental_bin(bin_max):
    bins = [1]
    alpha = 1
    for n in range(bin_max):
        new_val = bins[-1] + (n + 1)**alpha
        alpha += 0.02
        if new_val >= bin_max:
            bins.append(bin_max)
            break
        bins.append(new_val)
    return np.array(bins)


bar_figure_preamble(par)

# calculate colorbar colors:
log_par = np.log(par)
color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

n = len(board_counter)
x = np.arange(n) + 1
for agent in tqdm(estimators, desc='Plotting cumulative average loss'):
    y = model_losses[agent]
    # log-scaled bins
    bins = incremental_bin(len(y))
    widths = (bins[1:] - bins[:-1])
    x = np.arange(len(y)) + 1

    # Calculate histogram.
    # np.histogram counts how many elements of x fall in each bin.
    # by specifying 'weights', you can make it sum weights instead of counting.
    # divide by hist_count after to get an average of the weights of each bin.
    hist_count = np.histogram(x, bins=bins)
    hist_value = np.histogram(x, bins=bins, weights=y)
    # Divide sum to get average:
    hist_norm = hist_value[0] / hist_count[0]

    plt.bar(bins[:-1], hist_norm, widths)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('State rank', fontsize=font)
plt.ylabel('Loss', fontsize=font - 2)
plt.title('Value loss of fully-trained agent', fontsize=font)
plt.xticks(fontsize=font_num)
plt.yticks(fontsize=font_num)
plt.tight_layout()
name = 'value_loss'
plt.savefig('plots/'+name+'.png', dpi=900)


