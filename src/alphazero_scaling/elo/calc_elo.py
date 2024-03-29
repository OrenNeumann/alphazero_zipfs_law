from src.alphazero_scaling.elo.Bayesian_Elo import bayeselo
import numpy as np

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
    if matches.ndim != 2:
        raise ValueError('BayesElo only works for 2D matrices.')
    n_agents = len(matches)
    agents = list(map(str, list(range(n_agents))))
    r = bayeselo.ResultSet()
    print('Loading games...')
    for i in range(dims[0]):
        player_1 = i
        for j in range(dims[1]):
            player_2 = j
            p = matches[i, j]
            if p is np.ma.masked:
                continue
            if i == j:
                continue
            p = int(p * 400)
            for k in range(400):
                # just say we had 800 games and none of them tied:
                r.append(player_1, player_2, int(k < p) * 2)
                r.append(player_2, player_1, int(k > p) * 2)

    print('Calculating rating...')
    e = bayeselo.EloRating(r, agents)
    e.offset(1000)
    e.mm()
    e.exact_dist()
    print(e)
    x = e.__str__().split("\n")
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
