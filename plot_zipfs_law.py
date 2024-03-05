import numpy as np
import matplotlib.pyplot as plt
import yaml
from src.general.general_utils import fit_power_law

"""
Plot all Zipf's law figures used in the workshop abstact.
For plotting the toy-model power law, use plot_theory.py.
"""


env = 'pentago'#'connect4'
mode = 'train'
with open("config/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
path = config['paths']['plot_data']


w, h = plt.figaspect(0.7) # 0.6,0.7
plt.figure(2, figsize=(w, h))
plt.style.use(['grid'])

font = 18 -2 # 20
font_num = 16 -2 # 18
#ax = plt.gca()

if env == 'connect4':
    low = 5 * 10 ** 2  # lower fit bound
    up = 10 ** 5  # upper fit bound
else:
    low = 2 * 10 ** 3  # lower fit bound
    up = 2 * 10 ** 5  # upper fit bound

if mode == 'train':
    models = ['q_0_0', 'q_6_0']
    if env == 'connect4':
        names = [r'$6 \cdot 10^2$ parameters', r'$2.3 \cdot 10^5$ parameters']
    else:
        names = [r'$1.9 \cdot 10^3$ parameters', r'$2.9 \cdot 10^5$ parameters']
    fit_colors = ['olivedrab', 'dodgerblue']
    data_colors = ['navy', 'darkviolet']

    for i in range(2):
        model = models[i]
        load_path = path + env + '/train_' + model + '.npy'
        freq = np.load(load_path)

        # Fit a power-law
        print('fitting...')
        x_fit, y_fit, equation = fit_power_law(up, low, name=names[i], max_x=2 * 10 ** 6)
        x = np.arange(len(freq)) + 1

        print('plotting...')
        plt.scatter(x, freq, color=data_colors[i], s=2*40 / (10 + x), alpha=1)
        # plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation.format(c=c, m=m))
        plt.plot(x_fit, y_fit, color=fit_colors[i], linewidth=1.3, alpha=0.7, label=equation)

        if env == 'connect4':
            plt.title('Connect Four', fontsize=font)
        else:
            plt.title('Pentago', fontsize=font)

if mode == 'test':
    filename = 'q_6_0q_6_1_temp1'
    #filename = 'q_6_0q_6_1_temp0p25'
    #filename = 'random2'

    if filename == 'random2':
        low = 5 * 10 ** 2  # lower fit bound
        up = 4 * 10 ** 5  # upper fit bound
        plt.title('Random games, Connect Four', fontsize=font)
        n_points = 3 * 10 ** 6

        load_path = path + env + '/test_' + filename + '.npy'
        freq = np.load(load_path)

        # Fit a power-law
        x_fit, y_fit, equation = fit_power_law(up, low, max_x=n_points)
        x = np.arange(len(freq)) + 1
        plt.scatter(x, freq, color='dodgerblue', s=2*40 / (10 + x), alpha=1)
        # plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation.format(c=c, m=m))
        plt.plot(x_fit, y_fit, color='darkviolet', linewidth=1.3, alpha=0.7, label=equation)

    else: # plot two temperatures

        filename = 'q_6_0q_6_1_temp0p25'
        low = 5 * 10 ** 2  # lower fit bound
        up = 10 ** 4  # upper fit bound
        n_points = 6 * 10 ** 4
        name = r'$T=0.25$'
        load_path = '/mnt/ceph/neumann/AZ_new/count_states/plot_data/' + env + '/test_' + filename + '.npy'
        freq = np.load(load_path)

        # Fit a power-law
        x_fit, y_fit, equation = fit_power_law(up, low, name=name, max_x=n_points)
        x = np.arange(len(freq)) + 1
        plt.scatter(x, freq, color='navy', s=2*40 / (10 + x), alpha=1)
        # plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation.format(c=c, m=m))
        plt.plot(x_fit, y_fit, color='olivedrab', linewidth=1.3, alpha=0.7, label=equation)

        ####################################33

        filename = 'q_6_0q_6_1_temp1'
        low = 5 * 10 ** 2  # lower fit bound
        up = 4 * 10 ** 5  # upper fit bound
        name = r'$T=1$     '
        n_points = 6 * 10 ** 5
        load_path = path + env + '/test_' + filename + '.npy'
        freq = np.load(load_path)

        # Fit a power-law
        x_fit, y_fit, equation = fit_power_law(up, low, name=name, max_x=n_points)
        x = np.arange(len(freq)) + 1
        plt.scatter(x, freq, color='dodgerblue', s=2*40 / (10 + x), alpha=1)
        # plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation.format(c=c, m=m))
        plt.plot(x_fit, y_fit, color='darkviolet', linewidth=1.3, alpha=0.7, label=equation)




plt.ylabel('Frequency', fontsize=font)
plt.xlabel('Board state rank', fontsize=font)
plt.xscale('log')
plt.yscale('log')

plt.xticks(fontsize=font_num)
plt.yticks(fontsize=font_num)

plt.legend(fontsize=font - 3, framealpha=1)

plt.tight_layout()
plt.savefig('plots/myfig.png', dpi=900)
plt.show()
