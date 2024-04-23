import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


def _generate_temperature_zipf_curves(k):
    temps = np.array([0.07, 0.1, 0.14, 0.2, 0.25, 0.32, 0.45, 0.6, 0.8, 1, 1.4, 2, 3, 5])   
    max_q = 6
    n_copies = 3
    nets = []
    for i in range(max_q + 1):
        for j in range(n_copies):
            nets.append('q_' + str(i) + '_' + str(j))
    n = len(nets)
    counter = StateCounter(env='connect_four')
    freqs = dict()
    path_dir = f'../plot_data/temperature/game_data/temp_num_{k}'
    for pair in tqdm(list(combinations(range(n), 2)), desc=f'Collecting T={temps[k]} matches'): 
        path = path_dir + '/' + nets[pair[0]] + '_vs_' + nets[pair[1]] + '/'
        counter.collect_data(path=path, max_file_num=80, quiet=True, matches=True)
    freqs= np.array([item[1] for item in counter.frequencies.most_common()])
    with open(f'../plot_data/temperature/zipf_curves/temp_num_{k}.pkl', 'wb') as f:
        pickle.dump(freqs, f)


def plot_temperature_curves(load_data=True):
    temps = np.array([0.07, 0.1, 0.14, 0.2, 0.25, 0.32, 0.45, 0.6, 0.8, 1, 1.4, 2, 3, 5])
    log_t = np.log(temps)
    color_nums = (log_t - log_t.min()) / (log_t.max() - log_t.min()) 
    tf =12
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1.2, 1]})
    par = np.load('src/config/parameter_counts/connect_four.npy')
    zipf_exponents = np.zeros(len(temps))

    print('Plotting Connect Four Zipf curves at different temperatures.')
    if not load_data:
        for k in range(len(temps)):
            _generate_temperature_zipf_curves(k)
    for k,t in tqdm(list(enumerate(temps))[::-1], desc='Plotting Zipf curves'):
        with open(f'../plot_data/temperature/zipf_curves/temp_num_{k}.pkl', 'rb') as f:
            zipf_curve = pickle.load(f)
        ###
        zipf_curve = zipf_curve[:np.argmax(zipf_curve == 1)] #prune
        ###
        x = np.arange(len(zipf_curve))+1
        axs[0].scatter(x,zipf_curve, color=cm.plasma(color_nums[k]), s=40 / (10 + x))

        #fitting between counts 200 and 4:
        low = np.argmax(zipf_curve < 200)
        up = np.argmax(zipf_curve == 4)
        x_nums = np.arange(up)[low:]
        [m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(zipf_curve[low:up]), deg=1, w=2 / x_nums)
        zipf_exponents[k]=-m

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('State rank',fontsize=tf)
    axs[0].set_ylabel('Frequency',fontsize=tf)
    axs[0].tick_params(axis='both', which='major', labelsize=tf-2)
    aligned_title(axs[0], r"$\bf{a.}$ Zipf's law and Elo scaling", tf-2)

    ##############################################################
    print('Plotting size scaling at different temperatures')
    axin0 = axs[0].inset_axes([0.6, 0.6, 0.4, 0.4])
    elo_exponents = np.zeros(len(temps))

    for k,t in enumerate(temps):
        if k==0:#
            continue
        print(f'({k+1}/{len(temps)}) Temperature: {t}')
        r = BayesElo()
        agents = PlayerNums()
        matches = np.load(f'../matches/temperature_scaling/connect_four_temp_num_{k}/matrix.npy')
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
        all_scores = []
        all_params = []
        for i in range(sizes):
            elo_scores.append(np.mean([elo[f'q_{i}_{j},10000'] for j in range(copies)]) )
            elo_stds.append(np.std([elo[f'q_{i}_{j},10000'] for j in range(copies)]) )
            for j in range(copies):
                all_scores.append(elo[f'q_{i}_{j},10000'])
                all_params.append(par[i])
        elo_scores = np.array(elo_scores)
        # Set Elo score range
        elo_scores += 100 - elo_scores.min()
        axin0.errorbar(par, elo_scores, yerr=[elo_stds, elo_stds], fmt='-o', 
                    color=cm.plasma(color_nums[k]), linewidth=0.5)
        #fitting:
        [m, c] = np.polyfit(np.log10(all_params[:-2*copies]), all_scores[:-2*copies], 1)
        elo_exponents[k] = m/400
    axin0.set_xscale('log')
    axin0.set_xlabel('Neural-net parameters',fontsize=tf-2)
    axin0.set_ylabel('Elo',fontsize=tf-2)
    axin0.tick_params(axis='both', which='major', labelsize=tf-4)
    
    ##############################################################
    print('Plotting exponents relation')
    for k in range(len(temps)):
        if k==0:#
            continue
        if k > 5:
            continue
    axs[1].plot(zipf_exponents[1:6], elo_exponents[1:6],  
                    markersize=0, linestyle='--', color='black')
    axs[1].scatter(zipf_exponents[1:6], elo_exponents[1:6], c=cm.plasma(color_nums[1:6]), s=40)
    axs[1].set_xlabel('Zipf exponent',fontsize=tf)
    axs[1].set_ylabel('Elo exponent',fontsize=tf)
    axs[1].tick_params(axis='both', which='major', labelsize=tf-2)
    aligned_title(axs[1], r"$\bf{b.}$ Exponent correlation", tf-2)

    axin1 = axs[1].inset_axes([0.6, 0.1, 0.4, 0.4])
    axin1.scatter(zipf_exponents[1:], elo_exponents[1:], c=cm.plasma(color_nums[1:]), s=10)
    axin1.axvline(x=1, color='black', linestyle='--')
    axin1.tick_params(axis='both', which='major', labelsize=tf-4)
    axin1.annotate(r'$T=\infty$'+' \nZipf exponent', xy=(1, 0.5), xytext=(1.2, 0.5), arrowprops=dict(arrowstyle='->'))

    # Colorbar:
    norm = matplotlib.colors.LogNorm(vmin=temps.min(), vmax=temps.max())
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=norm)
    cbar = fig.colorbar(sm, ax=axs[1])
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Temperature', rotation=90, fontsize=tf)

    fig.tight_layout()
    print('Saving figure (can take a while)...')
    fig.savefig('./plots/temperature_curves.png', dpi=300)
