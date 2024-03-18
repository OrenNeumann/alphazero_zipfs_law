import matplotlib.pyplot as plt
import numpy as np
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_frequency.resign_counter import ResignationCounter, get_model
from src.general.general_utils import models_path
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import Figure

"""
Count board states played from actor logfiles of AlphaZero agents.
"""

# Choose game type:
game_num = 2
games = ['connect_four', 'pentago', 'oware', 'checkers']

env = games[game_num]
path = models_path()
data_paths = {'connect_four': 'connect_four_10000/q_0_0/',#'connect_four_10000/f_4_2/',
              'pentago': 'pentago_t5_10000/q_0_0/',
              'oware': 'oware_10000/q_2_0/',
              'checkers': 'checkers/q_6_0/'}
path += data_paths[env]
print('Collecting '+env+' games:')

file_num = 1

# Process all games
counter = ResignationCounter(env=env, model=get_model(path), save_serial=True,
                 save_turn_num=True)

counter.collect_data(path=path, max_file_num=file_num)


font = 18 - 2
font_num = 16 - 2

fig = Figure(x_label='State rank',text_font=font, number_font=font_num,fig_num=1)

def plot_turns(y, name, y_label):
    fig.preamble()
    plt.scatter(x, y, s=40 * 3 / (10 + x), alpha=1, color='green')
    plt.xscale('log')
    plt.yscale('log')
    fig.y_label = y_label
    fig.epilogue()
    fig.save(name)

print('Plot turns')
y = sort_by_frequency(data=counter.turns_played, counter=counter.frequencies)
plot_turns(y, name='turns_takenr_resignation', y_label='Turn num.')



print('Plot zipf distributions')
regular_counter = StateCounter(env=env, save_serial=True,
                 save_turn_num=True)
regular_counter.collect_data(path=path, max_file_num=file_num)

fig = Figure(x_label='State rank',
             y_label='Frequency',
             text_font=font,
             number_font=font_num,
             legend=True,
             fig_num=2)
fig.preamble()

# Sort by frequency
freq_regular = np.array([item[1] for item in regular_counter.frequencies.most_common()])
freq_resignation = np.array([item[1] for item in counter.frequencies.most_common()])

x = np.arange(len(freq_regular)) + 1
plt.scatter(x, freq_regular, s=40 / (10 + x), label='With resignation')
x = np.arange(len(freq_resignation)) + 1
plt.scatter(x, freq_resignation, s=40 / (10 + x), label='With resignation')

plt.xscale('log')
plt.yscale('log')
fig.epilogue()
fig.save('zipf_distribution_resignation')

