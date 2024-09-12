from src.alphazero_scaling.elo.Bayesian_Elo import bayeselo
import numpy as np


class PlayerNums(object):
    """ Map model,checkpoint pairs to unique numbers."""
    def __init__(self):
        self.n = 0
        self.d = dict()

    def add(self, model, checkpoint):
        player = model + ',' + str(checkpoint)
        if player not in self.d:
            self.d[player] = self.n
            self.n += 1

    def num(self, model, checkpoint):
        return self.d[model + ',' + str(checkpoint)]

    def names(self):
        """ return dict keys in order of values """
        return sorted(self.d, key=self.d.get)


class BayesElo(object):
    def __init__(self):
        self.rating = bayeselo.ResultSet()

    def add_match(self, n: int, m: int, p: float):
        for k in range(800):
            # 1600 games, no ties:
            self.rating.append(n, m, int(k < p * 800) * 2)
            self.rating.append(m, n, int(k > p * 800) * 2)
            
    def extract_elo(self, agents: PlayerNums):
        e = bayeselo.EloRating(self.rating, agents.names())
        e.offset(1000)
        e.mm()
        e.exact_dist()
        print(e)
        # Place players and scores in a dict
        x = str(e).split("\n")
        players = [row.split()[1] for row in x[1:-1]]
        scores = [int(row.split()[2]) for row in x[1:-1]]
        scores = np.array(scores) - min(scores)
        elos = dict(zip(players, scores))
        return elos


