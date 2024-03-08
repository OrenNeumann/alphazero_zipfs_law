import numpy as np
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import Figure
import matplotlib.pyplot as plt

"""
Plot turn related data: how late/early in the game do states appear.
"""

# Choose game type:
game_num = 2
games = ['connect_four', 'pentago', 'oware', 'checkers']

env = games[game_num]

data_labels = [0, 1, 2, 3, 4, 5]  # for oware no 6

state_counter = gather_data(env, data_labels, max_file_num=39, save_turn_num=True)
state_counter.prune_low_frequencies(4)
turns_played = state_counter.turns_played
turns_to_end = state_counter.turns_to_end
board_counter = state_counter.frequencies

# Turns analysis:
n = len(board_counter)
x = np.arange(n) + 1

font = 18 - 2
font_num = 16 - 2

fig = Figure(x_label='State rank',text_font=font, number_font=font_num)
#plt.xscale('log')

def plot_turns(y, name, y_label, y_logscale=False):
    fig.preamble()
    plt.scatter(x, y, s=40 * 3 / (10 + x), alpha=1, color='green')
    plt.xscale('log')
    if y_logscale:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    fig.y_label = y_label
    fig.epilogue()
    fig.save(name)
    plt.show()


print('plot turns taken so far')
y = sort_by_frequency(data=turns_played, counter=board_counter)
y = np.cumsum(y) / (np.arange(n) + 1)
plot_turns(y, name='turns_taken', y_label='Average turn num.')

print('plot turns until end of game')
y = sort_by_frequency(data=turns_to_end, counter=board_counter)
y = np.cumsum(y) / (np.arange(n) + 1)
plot_turns(y, name='turns_left', y_label='Average turns until end')

print('Plot raw data (to see U-shape)')
y = sort_by_frequency(data=turns_played, counter=board_counter)
# y = sort_by_frequency(data=turns_to_end, counter=board_counter)
plot_turns(y, name='turns_taken_raw', y_label='Turn num.', y_logscale=True)  # ylabel 'Turns left'
