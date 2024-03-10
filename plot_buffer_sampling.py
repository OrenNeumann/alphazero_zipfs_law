import matplotlib.pyplot as plt
import numpy as np
from src.data_analysis.state_frequency.buffer_counter import BufferCounter, UniqueBufferCounter
from src.general.general_utils import models_path, game_path, fit_power_law
from src.plotting.plot_utils import Figure

"""
Plot state frequency with a buffer.
"""

# Choose game type:
game_num = 0
games = ['connect_four', 'pentago', 'oware', 'checkers']
env = games[game_num]

path = models_path() + game_path(env)
model = 'q_6_2'
path += model + '/'

print('Collecting ' + env + ' games:')
num_files = 3

fit_colors = ['olivedrab', 'dodgerblue']
data_colors = ['navy', 'darkviolet']
fig = Figure(x_label='Board state number', 
             y_label='Frequency',
             title='Frequency of Board States, ' + env, 
             legend=True)
fig.preamble()

state_counter = BufferCounter(env=env, cut_early_games=False)
state_counter.collect_data(path, max_file_num=num_files)

# Sort by frequency
#board_freq = sorted(state_counter.frequencies.items(), key=lambda x: x[1], reverse=True)
freq = [item[1] for item in state_counter.frequencies.most_common()]

# Fit a power-law
if env == 'connect_four':
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
n_points = len(freq)  # 5 * 10 ** 6  # Display top n_points

x = np.arange(n_points) + 1
plt.scatter(x, freq[:n_points], color=data_colors[0], s=40 / (10 + x))
plt.plot(x_fit, y_fit, color=fit_colors[0], linewidth=1.5, label='Uniform sampling, '+equation)

####### run again to get unique-sampling data: ###############

state_counter = UniqueBufferCounter(env=env, cut_early_games=False)
state_counter.collect_data(path, max_file_num=num_files)

# Sort by frequency
#board_freq = sorted(state_counter.frequencies.items(), key=lambda x: x[1], reverse=True)
freq = [item[1] for item in state_counter.frequencies.most_common()]

# set fitting limits to capture the tail power-law:
# omit last two plateaus (freq=1 or 2)
up = len(freq) - sum(np.array(freq)<3)
low = int(up/10**2)
# plot from ~middle of x-axis on
min_x = 10**(np.log10(len(freq))/2 - 1)
max_x = len(freq) - sum(np.array(freq)==1)
x_fit, y_fit, equation = fit_power_law(freq, up_bound=up, low_bound=low, full_equation=True,
                                       min_x=min_x, max_x=max_x)

n_points = len(freq)  # 5 * 10 ** 6  # Display top n_points

x = np.arange(n_points) + 1
plt.scatter(x, freq[:n_points], color=data_colors[1], s=40 / (10 + x))
plt.plot(x_fit, y_fit, color=fit_colors[1], linewidth=1.5, label='Unique sampling, '+equation)

plt.xscale('log')
plt.yscale('log')

fig.epilogue()
fig.save('buffer_distribution')
plt.show()
