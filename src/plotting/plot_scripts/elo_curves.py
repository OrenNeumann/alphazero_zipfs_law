from math import pi
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


def _oware_size_scaling():
    """ Load match matrices and calculate Elo ratings. """
    checkpoints = [20, 30, 50, 70, 100, 150, 230, 340, 510, 770, 1150, 1730, 2590, 3880, 5820, 8730, 10000]
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
        for cp in checkpoints:
            agents.add(model, cp)
        if len(matches) != len(checkpoints):
            raise ValueError('Matrix size does not match number of checkpoints.')
        for i, j in combinations(range(len(matches)), 2):
            num_i = agents.num(model, checkpoints[i])
            num_j = agents.num(model, checkpoints[j])
            r.add_match(num_i, num_j, p=matches[i, j])

    ######## load fixed-checkpoint models ########
    def fc_model_ordering():
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

    for cp in tqdm(checkpoints, desc='Loading fixed-checkpoint matches'):
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
        counter = gather_data(env, data_labels, cutoff=50, max_file_num=14, save_turn_num=True)
        counter.prune_low_frequencies(10)
        freq_cutoff = np.array([item[1] for item in counter.frequencies.most_common()])

        print(f'Gathering data for {env} (no cutoff).')
        counter = gather_data(env, data_labels, max_file_num=10, save_turn_num=True)
        counter.prune_low_frequencies(10)
        freq_normal = np.array([item[1] for item in counter.frequencies.most_common()])

        with open(f'../plot_data/elo_curves/{env}_freqs.pkl', 'wb') as f:
            pickle.dump({'freq_cutoff': freq_cutoff, 'freq_normal': freq_normal}, f)


def plot_scaling_failure(load=True):
    """ plot oware and checkers elo curves."""
    tf =12
    l_width = 2
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    ax1 = axes[0]

    ######## plot checkers size scaling ########
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

    ######## plot oware size scaling ########
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
    ax1.set_xlabel('Neural net parameters',fontsize=tf)
    ax1.set_ylabel('Elo',fontsize=tf)
    ax1.tick_params(axis='both', which='major', labelsize=tf-2)
    ax1.legend(fontsize=tf-2)
    aligned_title(ax1, title=r'$\bf{a.}$ Scaling curves',font=tf+4)

    #################################################

    def zipf_law_plot(ax, freqs):
        x = np.arange(len(freqs['freq_cutoff']))
        ax.scatter(x,freqs['freq_cutoff'], color='dodgerblue', s=40 / (10 + np.log10(x)))
        xy = (5*10**3, freqs['freq_cutoff'][5*10**3])
        plt.annotate('Only early-game states', xy = xy, 
             fontsize = tf, xytext =[xy[0]/10,xy[1]/1.2])
             #arrowprops = dict(facecolor = 'red'),
             #color = 'g')
        x = np.arange(len(freqs['freq_normal']))
        ax.scatter(x,freqs['freq_normal'], color='gold', s=40 / (10 + np.log10(x)))
        xy = (10**4, freqs['freq_normal'][10**4])
        plt.annotate('All states', xy = xy, 
             fontsize = tf, xytext =[xy[0]/1.2,xy[1]*10])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('State rank',fontsize=tf)
        ax.set_ylabel('Frequency',fontsize=tf)
        ax.tick_params(axis='both', which='major', labelsize=tf-2)

    if not load:
        _bent_zipf_laws()

    ax2 = axes[1]
    with open('../plot_data/elo_curves/oware_freqs.pkl', 'rb') as f:
        oware_freqs = pickle.load(f)
    zipf_law_plot(ax2, oware_freqs)
    aligned_title(ax2, title=r'$\bf{b.}$ Oware Zipf\'s law',font=tf+4)
    ax3 = axes[2]
    with open('../plot_data/elo_curves/checkers_freqs.pkl', 'rb') as f:
        checkers_freqs = pickle.load(f)
    zipf_law_plot(ax3, checkers_freqs)
    aligned_title(ax3, title=r'$\bf{c.}$ Checkers Zipf\'s law',font=tf+4)

    fig.tight_layout()
    fig.savefig('plots/oware_checkers_scaling.png', dpi=300)
    