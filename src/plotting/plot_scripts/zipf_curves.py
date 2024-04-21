import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from tqdm import tqdm
from itertools import combinations
from src.plotting.plot_utils import aligned_title
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.gather_agent_data import gather_data
from src.general.general_utils import models_path, game_path
from src.alphazero_scaling.elo.utils import PlayerNums, BayesElo


plt.style.use(['science','nature','grid'])

def _generate_zipf_curves(env, models):
    print('Generating Zipf curves for ' + env)
    counter = StateCounter(env=env)
    freqs = dict()
    for model in models:
        print('Collecting ' + model + ' games:')
        path = models_path() + game_path(env) + model + '/'
        counter.collect_data(path=path, max_file_num=39)
        freqs[model] = np.array([item[1] for item in counter.frequencies.most_common()])
    print('Collecting mixed states:')
    counter = gather_data(env, labels=[0,2,4,6], max_file_num=10)
    freqs['combined'] = np.array([item[1] for item in counter.frequencies.most_common()])
    with open(f'../plot_data/zipf_curves/zipf_curves_{env}.pkl', 'wb') as f:
        pickle.dump(freqs, f)

def _fit_power_law(freq, ylabel, env):
    if env == 'connect_four' or env == 'pentago':
        low = 10**2
        up = 2*10**6
    else:
        low = np.argmax(freq < 10**2)#10**2
        up = np.argmax(freq < 10**1)#int(len(freq)/10**2)
    x_nums = np.arange(up)[low:]
    [m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(freq[low:up]), deg=1, w=2 / np.sqrt(x_nums))
    exp = str(round(-m, 2))
    equation = r'$\alpha = ' + exp + '$'
    if ylabel:
        equation = r'${\bf \alpha = ' + exp + '}$'
    x = np.arange(len(freq)) + 1
    y = 10 ** c * x[:int(10**7)] ** m
    bound = np.argmax(y < 1)
    x_fit = [1, bound+1]
    y_fit = [y[0], y[bound]]
    return x_fit, y_fit, equation


def _plot_curve(ax, y, env, xlabel=False, ylabel=False, tf=12):
    x = np.arange(len(y)) + 1
    ax.scatter(x, y, color='dodgerblue')
    x_fit, y_fit, equation = _fit_power_law(y, ylabel, env)
    ax.plot(x_fit, y_fit, color='black', linewidth=1.5, label=equation)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=tf-2)
    if xlabel:
        ax.set_xlabel('State rank',fontsize=tf)
    if ylabel:
        ax.set_ylabel('Frequency',fontsize=tf)
    ax.legend(fontsize=tf-2, loc='upper right')


def plot_zipf_curves(load_data=True):
    models = ['q_0_0', 'q_2_0', 'q_4_0', 'q_6_0']
    envs = ['connect_four', 'pentago', 'oware', 'checkers']
    titles = [r'${\bf Connect \, Four}$', r'${\bf Pentago}$', r'${\bf Oware}$', r'${\bf Checkers}$']
    pars = [np.load(f'src/config/parameter_counts/{env}.npy').take([0,2,4,6]) for env in envs]
    if not load_data:
        for env in envs:
            _generate_zipf_curves(env, models)

    tf =12
    # Create figure and subplots
    fig, axs = plt.subplots(nrows=len(envs), ncols=len(models)+1, figsize=(12, 4*3))

    for i, env in enumerate(tqdm(envs,desc='Plotting Zipf curves')):
        with open(f'../plot_data/zipf_curves/zipf_curves_{env}.pkl', 'rb') as f:
            zipf_curves = pickle.load(f)
        xlabel = (i==3)
        for j, model in enumerate(models):
            ax = axs[i, j+1]
            _plot_curve(ax, zipf_curves[model], env, xlabel=xlabel)
            size = str(round(np.log10(pars[i][j]), 1))
            aligned_title(ax, '$10^{'+size+'}$ parameters', tf-2)
        _plot_curve(axs[i, 0], zipf_curves['combined'], env, xlabel=xlabel, ylabel=True)
        aligned_title(axs[i,0], titles[i], tf)
    fig.tight_layout()
    print('Saving figure (can take a while)...')
    fig.savefig('plots/zipf_curves.png', dpi=300)


def plot_temperature_curves():
    temps = [0.07, 0.1, 0.14, 0.2, 0.25, 0.32, 0.45, 0.6, 0.8, 1, 1.4, 2, 3, 5]
    print('Plotting size scaling at different temperatures')
    tf =12
    l_width = 2
    par = np.load('src/config/parameter_counts/connect_four.npy')
    r = BayesElo()
    agents = PlayerNums()
    matches = np.load('../matches/checkers/matrix.npy')
    sizes = 7
    copies = 4
    for i, j in combinations(range(len(matches)), 2):
        r.add_match(i, j, p=matches[i, j])
    for i in range(sizes):
        for j in range(copies):
            agents.add(f'q_{i}_{j}', 10000)
    elo = r.extract_elo(agents)

    elo_scores = []
    elo_stds = []
    for i in range(sizes):
        elo_scores.append(np.mean([elo[f'q_{i}_{j},10000'] for j in range(copies)]) )
        elo_stds.append(np.std([elo[f'q_{i}_{j},10000'] for j in range(copies)]) )
    elo_scores = np.array(elo_scores)
    # Set Elo score range
    elo_scores += 100 - elo_scores.min()
    plt.errorbar(par, elo_scores, yerr=[elo_stds, elo_stds], fmt='-o', 
                 color='#2ca02c', linewidth=l_width, label='Oware (T-drop=50)')
    plt.plot(par, elo_scores, '-o', color='#bcbd22', linewidth=l_width, label='Checkers')

    print('Plotting Connect Four Zipf curves at different temperatures.')
    counter = StateCounter(env='connect_four')
    freqs = dict()
    for t in enumerate(temps):
        print('Collecting T=' + str(t) + ' games:')
        #path = models_path() + game_path(env) + model + '/'
        #counter.collect_data(path=path, max_file_num=39)
        #freqs[model] = np.array([item[1] for item in counter.frequencies.most_common()])
