import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from tqdm import tqdm
from src.plotting.plot_utils import aligned_title
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.gather_agent_data import gather_data
from src.general.general_utils import models_path, game_path


plt.style.use(['science','nature','grid'])

def _generate_zipf_curves(env: str, models: list[str]) -> None:
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

def _fit_power_law(freq, ylabel: bool, env: str) -> tuple[list, list, str]:
    if env in ['connect_four', 'pentago']:
        low = 10**2
        up = 2*10**6
    else:
        low = np.argmax(freq < 10**2)
        up = np.argmax(freq < 10**1)
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


def plot_main_zipf_curves(res: int = 300) -> None:
    print('~~~~~~~~~~~~~~~~~~~ Plotting Zipf curves (main paper) ~~~~~~~~~~~~~~~~~~~')
    envs = ['connect_four', 'pentago']
    colors = ['#377eb8', '#984ea3']
    tf =25
    fig = plt.figure(figsize=(12, 8))
    labels = []
    for i, env in enumerate(tqdm(envs,desc='Plotting Zipf curves')):
        with open(f'../plot_data/zipf_curves/zipf_curves_{env}.pkl', 'rb') as f:
            zipf_curves = pickle.load(f)
        y = zipf_curves['combined']
        x = np.arange(len(y)) + 1
        plt.scatter(x, y, color=colors[i], s=15)
        x_fit, y_fit, equation = _fit_power_law(y, True, env)
        labels.append(equation)
        plt.plot(x_fit, y_fit, color='black', linewidth=1.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=tf-2)
    plt.annotate('Connect Four: ' + labels[0], xy=(10**4, 4*10**2), xytext=(3*10**4, 8*10**3), arrowprops=dict(arrowstyle='->'), fontsize=tf)
    plt.annotate('Pentago: ' + labels[1], xy=(10**3, 4*10**2), xytext=(0.5*10**2, 4*10**1), arrowprops=dict(arrowstyle='->'), fontsize=tf)
    plt.xlabel('State rank',fontsize=tf)
    plt.ylabel('Frequency',fontsize=tf)
    fig.tight_layout()
    print('Saving figure (can take a while)...')
    fig.savefig('plots/main_zipf_curves.png', dpi=res)


def plot_appendix_zipf_curves(load_data: bool = True, res: int = 300) -> None:
    print('~~~~~~~~~~~~~~~~~~~ Plotting Zipf curves (appendix) ~~~~~~~~~~~~~~~~~~~')
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
    fig.savefig('plots/appendix_zipf_curves.png', dpi=res)
