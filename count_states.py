import matplotlib.pyplot as plt
import numpy as np
from data_analysis.game_data_analysis import process_games


"""
Count board states played from actor logfiles of AlphaZero agents.
"""

# Choose game type:
game_num = 0
games = ['connect4', 'pentago', 'oware', 'checkers']

env = games[game_num]
#path = '/mnt/ceph/neumann/alphazero/scratch_backup/models/'
path = '/home/oren/zipf/scratch_backup/models/connect_four_10000/'
data_paths = {'connect4': 'connect_four_10000/f_4_2',#'connect_four_10000/q_0_0',
              'pentago': 'pentago_t5_10000/q_0_0',
              'oware': 'oware_10000/q_1_0',
              'checkers': 'checkers/q_6_0'}
path += data_paths[env]
print('Collecting '+env+' games:')

# Process all games
# board_counter, _ = process_games(env, path)
board_counter, information = process_games(env, path, save_serial=True,
                                           save_turn_num=True, max_file_num=39)
serial_states = information['serials']
turn_numbers = information['turn_nums']

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
    low = 10**2
    up = int(len(freq)/10**2)
x_nums = np.arange(up)[low:]
[m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(freq[low:up]), deg=1, w=2 / x_nums)
equation = '10^{c:.2f} * n^{m:.2f}'

# Plot
plt.figure(figsize=(10, 6))
n_points = len(freq) #5 * 10 ** 6  # Display top n_points

#x_fit = np.array([1, n_points])
x_fit = np.array([low, up])
y_fit = 10 ** c * x_fit ** m

x = np.arange(n_points) + 1
plt.scatter(x, freq[:n_points], s=40 / (10 + x))
plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation.format(c=c, m=m))
plt.ylabel('Frequency')
plt.xlabel('Board state number')
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=n_points)
plt.title('Frequency of Board States')
plt.legend()
plt.tight_layout()
plt.savefig('plots/zipf_distribution.png', dpi=900)
plt.show()




