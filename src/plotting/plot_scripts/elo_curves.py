import pickle
from src.alphazero_scaling.elo.utils import PlayerNums, BayesElo
import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from src.plotting.plot_utils import aligned_title
from src.data_analysis.gather_agent_data import gather_data
import scienceplots

plt.style.use(['science','nature','grid'])

CHECKPOINTS = [20, 30, 50, 70, 100, 150, 230, 340, 510, 770, 1150, 1730, 2590, 3880, 5820, 8730, 10000]


def _oware_size_scaling() -> tuple[list[int], list[float], list[float], list[int], list[float], list[float]]:
    """ Load match matrices and calculate Elo ratings. """
    dir_name = '../matches/oware_base/'
    r = BayesElo()
    agents = PlayerNums()

    ######## load fixed-size models ########
    fixed_size_models = []
    # Enumerate self-matches models
    for i in range(6):
        for j in range(4):
            fixed_size_models.append('q_' + str(i) + '_' + str(j))
    
    for model in tqdm(fixed_size_models, desc='Loading fixed-size matches'):
        matches = np.load(dir_name + 'fixed_size/' + str(model) + '/matrix.npy')
        for cp in CHECKPOINTS:
            agents.add(model, cp)
        if len(matches) != len(CHECKPOINTS):
            raise ValueError('Matrix size does not match number of checkpoints.')
        for i, j in combinations(range(len(matches)), 2):
            num_i = agents.num(model, CHECKPOINTS[i])
            num_j = agents.num(model, CHECKPOINTS[j])
            r.add_match(num_i, num_j, p=matches[i, j])

    ######## load fixed-checkpoint models ########
    def fc_model_ordering() -> list[str]:
        # this misses q_6_3 and f_*_3 sadly
        max_q = 6
        min_f = 0
        max_f = 5
        n_copies = 3
        nets = []
        for i in range(max_q + 1):
            for j in range(n_copies):
                nets.append('q_' + str(i) + '_' + str(j))
            if min_f <= i <= max_f:
                for j in range(n_copies):
                    nets.append('f_' + str(i) + '_' + str(j))
        return nets
    
    fixed_checkpoint_models = fc_model_ordering()

    for cp in tqdm(CHECKPOINTS, desc='Loading fixed-checkpoint matches'):
        matches = np.load(dir_name + 'fixed_checkpoint/checkpoint_' + str(cp) + '/matrix.npy')
        for model in fixed_checkpoint_models:
            agents.add(model, cp)
        if len(matches) != len(fixed_checkpoint_models):
            raise ValueError('Matrix size does not match number of models.')
        for i, j in combinations(range(len(matches)), 2):
            num_i = agents.num(fixed_checkpoint_models[i], cp)
            num_j = agents.num(fixed_checkpoint_models[j], cp)
            r.add_match(num_i, num_j, p=matches[i, j])

    elo = r.extract_elo(agents)

    ######## plot oware size scaling ########
    par = np.array([155, 265, 399, 739, 1175, 2335, 3879, 8119, 13895, 30055, 52359, 115399, 203015])

    q_scores, f_scores = [], []
    q_sizes, q_error, q_means = [], [], []
    f_sizes, f_error, f_means = [], [], []
    i= 0
    for size in range(6): # Ignore last size, was trained with different temp. drop
        y=[]
        for copy in range(6):
            model = 'q_' + str(size) + '_' + str(copy) +',10000'
            if model in elo:
                q_scores.append(elo[model])
                y.append(elo[model])
        q_error.append(np.std(y))
        q_means.append(np.mean(y))
        q_sizes.append(par[i])
        i += 1
        y = []
        for copy in range(4):
            model = 'f_' + str(size) + '_' + str(copy) +',10000'
            if model in elo:
                f_scores.append(elo[model])
                y.append(elo[model])
        f_error.append(np.std(y))
        f_means.append(np.mean(y))
        f_sizes.append(par[i])
        i += 1
    return q_sizes, q_means, q_error, f_sizes, f_means, f_error


def _bent_zipf_laws():
    """ Clauculate state frequency distribution with and without turn cutoff."""
    data_labels = [0, 1, 2, 3, 4, 5, 6]
    for env in ['oware', 'checkers']:
        print(f'Gathering data for {env} (cutoff=40).')
        counter = gather_data(env, data_labels, cutoff=50, max_file_num=40, save_turn_num=True)
        counter.prune_low_frequencies(2)
        freq_cutoff = np.array([item[1] for item in counter.frequencies.most_common()])

        with open(f'../plot_data/elo_curves/{env}_freq_cutoff.pkl', 'wb') as f:
            pickle.dump(freq_cutoff, f)
        del counter
        del freq_cutoff

        print(f'Gathering data for {env} (no cutoff).')
        counter = gather_data(env, data_labels, max_file_num=40)
        counter.prune_low_frequencies(2)
        freq_normal = np.array([item[1] for item in counter.frequencies.most_common()])
        with open(f'../plot_data/elo_curves/{env}_freq_normal.pkl', 'wb') as f:
            pickle.dump(freq_normal, f)
        del counter
        del freq_normal


def plot_scaling_failure(load_data=True, res=300):
    """ plot oware and checkers elo curves."""
    print('~~~~~~~~~~~~~~~~~~~ Plotting scaling-failure plot ~~~~~~~~~~~~~~~~~~~')
    tf =12
    l_width = 2
    # Create figure and subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1, 1]})
    ax1 = axs[0]

    print('Plotting checkers size scaling')
    par = np.load('src/config/parameter_counts/checkers.npy')
    r = BayesElo()
    agents = PlayerNums()
    matches = np.load('../matches/checkers/matrix.npy')
    for i, j in combinations(range(len(matches)), 2):
        r.add_match(i, j, p=matches[i, j])
    for i in range(len(matches)):
        agents.add(f'q_{i}_0', 10000)
    elo = r.extract_elo(agents)
    elo_scores = np.array([elo[f'q_{i}_0,10000'] for i in range(len(matches))])
    # Set Elo score range
    elo_scores -= elo_scores.min() -100
    ax1.plot(par, elo_scores, '-o', color='#bcbd22', linewidth=l_width, label='Checkers')

    print('Plotting oware size scaling')
    q_sizes, q_means, q_error, f_sizes, f_means, f_error= _oware_size_scaling()
    # Set Elo score range
    min_elo = min(q_means) -100
    q_means = np.array(q_means) - min_elo
    f_means = np.array(f_means) - min_elo
    ax1.errorbar(q_sizes, q_means, yerr=[q_error, q_error], fmt='-o', 
                 color='#2ca02c', linewidth=l_width, label='Oware (T-drop=50)')
    ax1.errorbar(f_sizes, f_means, yerr=[f_error, f_error], fmt='-o', 
                 color='limegreen', linewidth=l_width, label='Oware (T-drop=15)')

    ax1.set_xscale('log')
    ax1.set_xlabel('Neural-net parameters',fontsize=tf)
    ax1.set_ylabel('Elo',fontsize=tf)
    ax1.tick_params(axis='both', which='major', labelsize=tf-2)
    ax1.legend(fontsize=tf-2, loc='upper left', framealpha=0.5)
    aligned_title(ax1, title=r'$\bf{A.}$ Scaling curves',font=tf+4)

    #################################################
    print('Plotting Zipf\'s law curves')

    def zipf_law_plot(ax, freqs):
        x = np.arange(len(freqs['freq_cutoff']))+1
        ax.scatter(x,freqs['freq_cutoff'], color='gold', s=40 / (10 + np.log10(x)), label='No late-game states')
        x = np.arange(len(freqs['freq_normal']))+1
        ax.scatter(x,freqs['freq_normal'], color='dodgerblue', s=40 / (10 + np.log10(x)), label='All states')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('State rank',fontsize=tf)
        ax.set_ylabel('Frequency',fontsize=tf)
        ax.legend(fontsize=tf-2, loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=tf-2)

    if not load_data:
        _bent_zipf_laws()

    for i, env in enumerate(['oware', 'checkers']):
        ax = axs[i+1]
        with open(f'../plot_data/elo_curves/{env}_freq_cutoff.pkl', 'rb') as f:
            cutoff_freqs = pickle.load(f)
        with open(f'../plot_data/elo_curves/{env}_freq_normal.pkl', 'rb') as f:
            normal_freqs = pickle.load(f)
        zipf_law_plot(ax, {'freq_cutoff': cutoff_freqs, 'freq_normal': normal_freqs})
        if env == 'oware':
            aligned_title(ax, title=r"$\bf{B.}$ Oware Zipf's law",font=tf+4)
        else:
            aligned_title(ax, title=r"$\bf{C.}$ Checkers Zipf's law",font=tf+4)

    fig.tight_layout()
    fig.savefig('plots/oware_checkers_scaling.png', dpi=res)
    