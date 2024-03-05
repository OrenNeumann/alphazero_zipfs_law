import numpy as np
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.utils import get_model_value_estimator, models_path
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

# from multiprocessing import Pool

"""
Value loss
"""

# Choose game type:
game_num = 0
games = ['connect4', 'pentago', 'oware', 'checkers']
env = games[game_num]
path = models_path()

num_processes = 20


def multiprocess_values(data):
    return model_values(data)


estimators = [0, 1, 2, 3, 4, 5, 6]  # for oware no 6
model_losses = dict()
for agent in estimators:
    data_labels = [agent]
    board_counter, info = gather_data(env, data_labels, max_file_num=10, save_serial=True, save_value=True)
    serials = info['serials']
    real_values = info['values']

    # (check) seems pruning 1's reduces by one OOM, 2's and 3's together by another OOM.
    print('Counter length before pruning:', len(board_counter))
    board_counter = Counter({k: c for k, c in board_counter.items() if c >= 2})
    print('Counter length after pruning: ', len(board_counter))

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

    model_losses[agent] = (z - v) ** 2

par = np.load('config/parameter_counts/' + env + '.npy')

#### colorbar plot cargo-cult code ###
w, h = plt.figaspect(0.6)
plt.figure(2, figsize=(w, h))
plt.style.use(['grid'])
font = 18
font_num = 16
plt.clf()
ax = plt.gca()
norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
# create a scalarmappable from the colormap
sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
cbar = plt.colorbar(sm)
cbar.ax.tick_params(labelsize=font_num)
cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=font)
#################################

# calculate colorbar colors:
log_par = np.log(par)
color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

for agent in tqdm(estimators, desc='Plotting cumulative average loss'):
    y = model_losses[agent]
    # plotting cumulative average of loss
    x = np.arange(len(y)) + 1
    y = np.cumsum(y) / x

    # weighted average with frequency:
    # y = np.cumsum(np.array(y_losses)*freq) / np.cumsum(freq)

    plt.scatter(x, y, s=40 / (10 + x), alpha=0.3, color=cm.viridis(color_nums[agent]))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('State rank', fontsize=font)
plt.ylabel('Cumulative average of the loss', fontsize=font - 2)
plt.title('Value loss on the train set', fontsize=font)
plt.xticks(fontsize=font_num)
plt.yticks(fontsize=font_num)
plt.tight_layout()
name = 'value_loss'
plt.savefig('plots/' + name + '.png', dpi=900)
