from src.alphazero_scaling.elo.Bayesian_Elo import bayeselo
import numpy as np
from itertools import combinations

"""
How this code works:
Feed in the games to 'r' in the form of:
r.append(x, y, z)
where:
x = number of first player
y = number of second player
z = game outcome: 0 loss, 1 tie, 2 win (for player x)

Then add player names to the EloRating function.

See original repository:
https://github.com/yytdfc/Bayesian-Elo
"""


def get_bayeselo(matches):
    dims = matches.shape
    if matches.ndim != 2 or dims[0] != dims[1]:
        raise ValueError('BayesElo only works for square 2D matrices.')
    n = len(matches)
    r = bayeselo.ResultSet()
    print('Loading games...')
    for player_1,player_2 in combinations(range(n), 2):
        p = matches[player_1, player_2]
        if p is np.ma.masked:
            continue
        for k in range(800):
            # just say we had 1600 games and none of them tied:
            r.append(player_1, player_2, int(k < p*800) * 2)
            r.append(player_2, player_1, int(k > p*800) * 2)
    print('Calculating rating...')
    agent_names = [str(i) for i in range(n)]
    e = bayeselo.EloRating(r, agent_names)
    e.offset(1000)
    e.mm()
    e.exact_dist()
    print(e)
    return _extract_elo(e)


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