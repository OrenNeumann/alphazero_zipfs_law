import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from src.plotting.plot_utils import aligned_title
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.gather_agent_data import gather_data
from src.general.general_utils import models_path, game_path

plt.style.use(['science','nature','grid'])

def _generate_zipf_curves(env, models):
    print('Generating Zipf curves for ' + env)
    counter = StateCounter(env=env)
    freqs = dict()
    for model in models:
        path = models_path() + game_path(env) + model + '/'
        counter.collect_data(path=path, max_file_num=39)
        freqs[model] = np.array([item[1] for item in counter.frequencies.most_common()])
    counter = gather_data(env, labels=[0,2,4,6], max_file_num=10)
    freqs['combined'] = np.array([item[1] for item in counter.frequencies.most_common()])
    with open(f'../plot_data/zipf_curves/zipf_curves_{env}.pkl', 'wb') as f:
        pickle.dump(freqs, f)

def _fit_power_law(freq):
    low = 10**2
    up = int(len(freq)/10**2)
    x_nums = np.arange(up)[low:]
    [m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(freq[low:up]), deg=1, w=2 / x_nums)
    exp = str(round(-m, 2))
    equation = r'$\alpha = ' + exp + '$'
    x = np.arange(len(freq)) + 1
    y = 10 ** c * x[:int(10**7)] ** m
    bound = np.argmax(y_fit < 1)
    x_fit = [1, bound+1]
    y_fit = [y[0], y[bound]]
    return x_fit, y_fit, equation


def _plot_curve(ax, y, par, tf=12):
    x = np.arange(len(y)) + 1
    ax.scatter(x, y, colot='dodgerblue')#, s=40 / (10 + x))
    x_fit, y_fit, equation = _fit_power_law(y)
    ax.plot(x_fit, y_fit, color='black', linewidth=1.5, label=equation)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=tf-2)
    ax.set_xlabel('State rank',fontsize=tf)
    ax.set_ylabel('Frequency',fontsize=tf)
    ax.legend(fontsize=tf-2)
    size = str(round(np.log10(par), 1))
    aligned_title(ax, f'$10^{size}$ parameters', tf)


def plot_zipf_curves(load_data=True):
    models = ['q_0_0', 'q_2_0', 'q_4_0', 'q_6_0']
    envs = ['connect_four', 'pentago', 'oware', 'checkers']
    pars = [np.load(f'src/config/parameter_counts/{env}.npy').take([0,2,4,6]) for env in envs]
    if not load_data:
        for env in envs:
            _generate_zipf_curves(env, models)

    tf =12
    # Create figure and subplots
    fig, axs = plt.subplots(nrows=len(envs), ncols=len(models)+1, figsize=(12, 4*3))

    print('Plotting Zipf curves')
    for i, env in enumerate(envs):
        with open(f'../plot_data/zipf_curves/zipf_curves_{env}.pkl', 'rb') as f:
            zipf_curves = pickle.load(f)
        for j, model in enumerate(models):
            ax = axs[i, j+1]
            y = zipf_curves[model]
            _plot_curve(ax, y, pars[i][j])
        aligned_title(axs[i,0], env, tf)

    fig.tight_layout()
    fig.savefig('plots/zipf_curves.png', dpi=300)