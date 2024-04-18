from src.alphazero_scaling.elo.utils import PlayerNums, BayesElo
import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from src.plotting.plot_utils import aligned_title
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

    i= 0
    q_scores = []
    f_scores = []
    q_sizes = []
    q_sizes = []
    q_error=[]
    q_means=[]
    f_sizes = []
    f_error=[]
    f_means=[]
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


def plot_scaling_failure():
    """ plot oware and checkers elo curves."""
    tf =12
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
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
    ax1.plot(par, [elo[f'q_{i}_0,10000'] for i in range(len(matches))], '-o', color='#bcbd22', label='Checkers')

    ######## plot oware size scaling ########
    print('Plotting oware size scaling')
    q_sizes, q_means, q_error, f_sizes, f_means, f_error= _oware_size_scaling()
    plt.errorbar(q_sizes, q_means, yerr=[q_error, q_error], fmt='-o', color='#2ca02c', label='Oware (T-drop=50')
    plt.errorbar(f_sizes, f_means, yerr=[f_error, f_error], fmt='-o', color='#238b23', label='Oware (T-drop=15')

    ax1.set_xscale('log')
    ax1.set_xlabel('Neural net parameters',fontsize=tf)
    ax1.set_ylabel('Elo',fontsize=tf)
    ax1.tick_params(axis='both', which='major', labelsize=tf-2)
    ax1.legend(fontsize=tf-2)
    aligned_title(ax1, title=r'$\bf{a.}$ Scaling curves',font=tf+4)
    plt.tight_layout()
    fig.savefig('plots/oware_checkers_scaling.png', dpi=300)
    