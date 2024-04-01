from src.alphazero_scaling.elo.utils import PlayerNums, BayesElo
import numpy as np
from tqdm import tqdm
from itertools import combinations
from src.plotting.plot_utils import Figure
import matplotlib.pyplot as plt

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

# this assumes 1) that the matrix is symmetric 2) len(matches) = len(checkpoints)
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

# assumes len(matches) = len(fixed_checkpoint_models)
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

######## Extract Elo ratings ########


elo = r.extract_elo(agents)

######## plot oware size scaling ########

par = np.array([155, 265, 399, 739, 1175, 2335, 3879, 8119, 13895, 30055, 52359, 115399, 203015])

font = 18 - 2
font_num = 16 - 2

fig = Figure(x_label='Neural net parameters', 
            y_label='Elo', 
            title='Oware size scaling',
            text_font=font, 
            number_font=font_num)
fig.preamble()

i= 0
scores = []
sizes = []
for size in range(7):
    for copy in range(4):
        model = 'q_' + str(size) + '_' + str(copy) +',10000'
        if model in elo:
            scores.append(elo[model])
            sizes.append(par[i])
    i += 1
    if size != 7:
        for copy in range(4):
            model = 'f_' + str(size) + '_' + str(copy) +',10000'
            if model in elo:
                scores.append(elo[model])
                sizes.append(par[i])
        i += 1
plt.scatter(sizes, scores)
plt.xscale('log')
fig.epilogue()
fig.save('oware_size_scaling')

