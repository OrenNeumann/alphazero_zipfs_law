import numpy as np
import pickle
from data_analysis.game_data_analysis import process_games, get_value_estimators
import collections
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
IMPORTANT:
I changed the way values are saved, or calculated by agents. Now an agent returns the value
according to the first player (player 0). The solver calculates the value for the current player.
Change the solver code to get the player 0 value.

ALSO:
I changed the model evaluator function to work with data chunks rather than individual states.

"""

env = 'connect4'

data_labels = [0, 2, 4, 6]
solver_values = dict()
board_counter = collections.Counter()
serial_states = dict()
for label in data_labels:
    num = str(label)
    path = '/mnt/ceph/neumann/alphazero/scratch_backup/models/connect_four_10000/q_' + num + '_0/log-actor'
    temp_counter, temp_info = process_games(env, path, save_serial=True, max_file_num=1)
    temp_serials = temp_info['serials']
    # add counts to the counter, and update new serial states:
    board_counter.update(temp_counter)
    serial_states.update(temp_serials)

    with open('solver_values_' + num + '0_1.pkl', "rb") as input_file:
        solver_values.update(pickle.load(input_file))

# sort board states:
a = board_counter.most_common()
freq = np.array([item[1] for item in a])


# Plot
def plot_zipfs_law(freq_vec):
    plt.figure(figsize=(10, 6))
    n_points = len(freq_vec)  # 5 * 10 ** 6  # Display top n_points
    plt.scatter(np.arange(n_points) + 1, freq_vec[:n_points], s=4, alpha=0.3)
    plt.ylabel('Frequency')
    plt.xlabel('Board state number')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(right=n_points)
    plt.title('Frequency of Board States')
    plt.tight_layout()
    plt.show()


plot_zipfs_law(freq)

#with open('model_values_1.pkl', "rb") as input_file:
#    model_values = pickle.load(input_file)

# Value loss analysis:
print('loss part...')
estimators = [0, 1, 2, 3, 4, 5, 6]
model_values = list()
# losses = dict()
for i in estimators:
    path_model = '/mnt/ceph/neumann/alphazero/scratch_backup/models/connect_four_10000/q_' + str(i) + '_0/'
    _, model_value = get_value_estimators(env, path_model)
    values_dict = dict()
    # temp_losses = dict()
    for key, serial in tqdm(serial_states.items(), desc="Estimating model state values"):
        values_dict[key] = model_value(serial)
        # temp_losses[key] = (solver_values[key] - temp_values[key]) ** 2
    model_values.append(values_dict)

n = len(board_counter)
par = np.array([608, 1304, 2984, 7496, 21128, 66824, 231944])

#### compute plot cargo code ###
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

x = np.arange(n) + 1
for i in tqdm(estimators, desc='Plotting cumulative average loss'):
    y_losses = []
    # sort values by descending frequency:
    for entry in board_counter.most_common():
        key = entry[0]
        y_losses.append((solver_values[key] - model_values[i][key]) ** 2)
    y_losses = np.array(y_losses)
    # plotting cumulative average of loss
    y = np.cumsum(y_losses) / x

    # standad deviation:
    #std = np.sqrt((y_losses**2).cumsum()/x - y**2)

    # weighted average with frequency:
    #y = np.cumsum(np.array(y_losses)*freq) / np.cumsum(freq)

    plt.scatter(x, y, s=40 / (10 + x), alpha=0.3, color=cm.viridis(color_nums[i]))
plt.xscale('log')
plt.xlabel('State rank', fontsize=font)
plt.ylabel('Cumulative average of the loss', fontsize=font - 2)
plt.title('Value loss of fully-trained agents', fontsize=font)
plt.xticks(fontsize=font_num)
plt.yticks(fontsize=font_num)
plt.tight_layout()


"""
# pretty zipf's law figure
w, h = plt.figaspect(0.6)
plt.figure(2, figsize=(w, h))
plt.style.use(['grid'])
font = 18
font_num = 16
plt.clf()
ax = plt.gca()
n_points = len(freq)  # 5 * 10 ** 6  # Display top n_points
plt.scatter(np.arange(n_points) + 1, freq[:n_points], s=40*3 / (10 + np.arange(n_points)), alpha=0.3)
plt.ylabel('Frequency', fontsize=font)
plt.xlabel('State rank', fontsize=font)
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=n_points)
plt.title('Dataset Zipf\'s law', fontsize=font)
plt.xticks(fontsize=font_num)
plt.yticks(fontsize=font_num)
plt.tight_layout()
plt.show()

"""


"""
#### compute plot cargo code ###
w, h = plt.figaspect(0.7)
plt.figure(2, figsize=(w, h))
plt.style.use(['grid'])
font = 18-2
font_num = 16-2
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
x = np.arange(n) + 1
for i in tqdm(estimators, desc='Plotting cumulative average loss'):
    y_losses = []
    # sort values by descending frequency:
    for entry in board_counter.most_common():
        key = entry[0]
        y_losses.append((solver_values[key] - model_values[i][key]) ** 2)
    y_losses = np.array(y_losses)
    # plotting cumulative average of loss
    y = np.cumsum(np.array(y_losses)*freq) / np.cumsum(freq)
    plt.scatter(x, y, s=40 / (10 + x), alpha=1, color=cm.viridis(color_nums[i]))
plt.xscale('log')
plt.xlabel('Board state rank', fontsize=font)
plt.ylabel('Weighted average value loss', fontsize=font - 2)
#plt.title('Value loss of fully-trained agents', fontsize=font)
plt.xticks(fontsize=font_num)
plt.yticks(fontsize=font_num)
plt.tight_layout()




"""