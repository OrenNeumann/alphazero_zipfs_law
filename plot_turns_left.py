import numpy as np
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import figure_preamble
import matplotlib.pyplot as plt

"""
Plot turn related data: how late/early in the game do states appear.
"""

# Choose game type:
game_num = 3
games = ['connect4', 'pentago', 'oware', 'checkers']

env = games[game_num]

data_labels = [0, 1, 2, 3, 4, 5]  # for oware no 6

board_counter, info = gather_data(env, data_labels, save_turn_num=True)
turns_played = info['turns_played']
turns_to_end = info['turns_to_end']

# Turns analysis:
n = len(board_counter)
x = np.arange(n) + 1


def plot_turns(y, name, y_label, y_logscale=False, fignum=1):
    figure_preamble(fig_num=fignum)
    font = 18 - 2
    font_num = 16 - 2
    plt.scatter(x, y, s=40 * 3 / (10 + x), alpha=1, color='green')
    plt.xscale('log')
    if y_logscale:
        plt.yscale('log')
    plt.xlabel('State rank', fontsize=font)
    plt.ylabel(y_label, fontsize=font - 2)
    plt.xticks(fontsize=font_num)
    plt.yticks(fontsize=font_num)
    plt.tight_layout()
    plt.savefig('plots/' + name + '.png', dpi=900)
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
