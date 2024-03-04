import matplotlib.pyplot as plt
import numpy as np
from data_analysis.game_data_analysis import process_games_with_buffer
from data_analysis.utils import models_path, fit_power_law, fit_logaritm

"""
Plot state frequency with a buffer.
"""

# Choose game type:
game_num = 0
games = ['connect4', 'pentago', 'oware', 'checkers']

env = games[game_num]
# path = '/mnt/ceph/neumann/alphazero/scratch_backup/models/'
# path = '/home/oren/zipf/scratch_backup/models/connect_four_10000/'
path = models_path()
data_paths = {'connect4': 'connect_four_10000/q_0_0',  # 'connect_four_10000/f_4_2',
              'pentago': 'pentago_t5_10000/q_0_0',
              'oware': 'oware_10000/q_1_0',
              'checkers': 'checkers/q_6_0'}
path += data_paths[env]
print('Collecting ' + env + ' games:')

num_files = 39
fit_colors = ['olivedrab', 'dodgerblue']
data_colors = ['navy', 'darkviolet']
# Process all games
# board_counter, information = process_games(env, path, save_serial=True,
#                                           max_file_num=39)
board_counter, _ = process_games_with_buffer(env, path, sample_unique_states=False, max_file_num=num_files)

# Sort by frequency
board_freq = sorted(board_counter.items(), key=lambda x: x[1], reverse=True)

# Extract the keys and the frequencies
# keys = [item[0] for item in board_freq]
freq = [item[1] for item in board_freq]

# Fit a power-law
if env == 'connect4':
    low = 5 * 10 ** 2  # lower fit bound
    up = 10 ** 5  # upper fit bound
elif env == 'pentago':
    low = 2 * 10 ** 3  # lower fit bound
    up = 2 * 10 ** 5  # upper fit bound
else:
    low = 10 ** 2
    up = int(len(freq) / 10 ** 2)
x_fit, y_fit, equation = fit_power_law(freq, up, low, full_equation=True)

# Plot
plt.figure(figsize=(10, 6))
n_points = len(freq)  # 5 * 10 ** 6  # Display top n_points

x = np.arange(n_points) + 1
plt.scatter(x, freq[:n_points], color=data_colors[0], s=40 / (10 + x))
plt.plot(x_fit, y_fit, color=fit_colors[0], linewidth=1.5, label=equation)

####### run again to get unique-sampling data: ###############


board_counter, _ = process_games_with_buffer(env, path, sample_unique_states=True, max_file_num=num_files)

# Sort by frequency
board_freq = sorted(board_counter.items(), key=lambda x: x[1], reverse=True)

# Extract the keys and the frequencies
# keys = [item[0] for item in board_freq]
freq = [item[1] for item in board_freq]

# Fit a power-law
if env == 'connect4':
    low = 5 * 10 ** 2  # lower fit bound
    up = 10 ** 5  # upper fit bound
elif env == 'pentago':
    low = 2 * 10 ** 3  # lower fit bound
    up = 2 * 10 ** 5  # upper fit bound
else:
    low = 10 ** 2
    up = int(len(freq) / 10 ** 2)
x_fit, y_fit, equation = fit_logaritm(freq, up, low)

n_points = len(freq)  # 5 * 10 ** 6  # Display top n_points

x = np.arange(n_points) + 1
plt.scatter(x, freq[:n_points], color=data_colors[1], s=40 / (10 + x))
plt.plot(x_fit, y_fit, color=fit_colors[1], linewidth=1.5, label=equation)
plt.ylabel('Frequency')
plt.xlabel('Board state number')
plt.xscale('log')
plt.yscale('log')
# plt.xlim(right=n_points)
plt.title('Frequency of Board States, ' + env)
plt.legend()
plt.tight_layout()
plt.savefig('plots/buffer_distribution.png', dpi=900)
plt.show()
