from src.alphazero_scaling.elo.Bayesian_Elo import bayeselo
import numpy as np
from tqdm import tqdm
from itertools import combinations

""" Load match matrices and calculate Elo ratings. """

checkpoints = [20, 30, 50, 70, 100, 150, 230, 340, 510, 770, 1150, 1730, 2590, 3880, 5820, 8730, 10000]
dir_name = '../matches/oware_base/'
r = bayeselo.ResultSet()

def add_match(n,m,p):
    print(n,m,p)
    print(type(n), type(m), type(p))
    for k in range(800):
        # just say we had 1600 games and none of them tied:
        r.append(n, m, int(k < p*800) * 2)
        r.append(m, n, int(k > p*800) * 2)

class PlayerNums(object):
    def __init__(self):
        self.n = 0
        self.d = dict()

    def add(self, model, checkpoint):
        player = model+'/'+str(checkpoint)
        if player not in self.d:
            self.d[player] = self.num
            self.n += 1
    
    def num(self, model, checkpoint):
        return self.d[model+'/'+str(checkpoint)]
    
    def names(self):
        """ return dict keys in order of values """
        return sorted(self.d, key=self.d.get)


agents = PlayerNums()

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
    for i,j in combinations(range(len(matches)), 2):
        p = matches[i, j]
        num_i = agents.num(model, checkpoints[i])
        num_j = agents.num(model, checkpoints[j])
        add_match(num_i, num_j, p)


fixed_checkpoint_models = []
# this misses q_6_3 and f_*_3 sadly
for i in range(7):
    for j in range(3):
        fixed_checkpoint_models.append('q_' + str(i) + '_' + str(j))
for i in range(6):
    for j in range(3):       
        fixed_checkpoint_models.append('f_' + str(i) + '_' + str(j))


# assumes len(matches) = len(fixed_checkpoint_models)
for cp in tqdm(checkpoints, desc='Loading fixed-checkpoint matches'):
    matches = np.load(dir_name + 'fixed_checkpoint/checkpoint_' + str(cp) + '/matrix.npy')
    for model in fixed_checkpoint_models:
        agents.add(model, cp)
    for i,j in combinations(range(len(matches)), 2):
        p = matches[i, j]
        num_i = agents.num(fixed_checkpoint_models[i], cp)
        num_j = agents.num(fixed_checkpoint_models[j], cp)
        add_match(num_i, num_j, p)


e = bayeselo.EloRating(r, agents.names())
e.offset(1000)
e.mm()
e.exact_dist()
print(e)

def _extract_elo(elo_rating):
    x = elo_rating.__str__().split("\n")
    table = []
    for row in x:
        table.append(row.split())
    table = np.array(table[:-1])
    agent_order = table[:, 1]
    elo = table[:, 2]
    agent_order = agent_order[1:].astype(int)
    elo = elo[1:].astype(float)
    elo = elo[agent_order.argsort()]
    return elo

elo = _extract_elo(e)