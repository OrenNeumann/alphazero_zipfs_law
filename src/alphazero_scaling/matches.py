"""
Play matches between 2 trained models and see how they compare to each other
This code produces a full matrix of matches between all trained nets.
Pairs played are: all combinations sharing a copy number, and all combinations sharing a size number.
Temperature drop is currently set to infinity, meaning temperature is constant throughout the game.
"""
from absl import app
import numpy as np
from itertools import combinations
import os
from shutil import copyfile

from open_spiel.python.utils import spawn

import AZ_helper_lib as AZh


# Name of the directory where the models are saved. Don't forget to change 'game' in config.
dir_name = 'oware'
# Save all logs to:
path_logs = '/scratch/compmatsc/neumann/matches/zipf_law/' + dir_name + '_ur/'
# Create the directory, then copy the config.json file into it: (otherwise crashes)
if not os.path.exists(path_logs):
    os.makedirs(path_logs)
    copyfile('/scratch/compmatsc/neumann/matches/config.json', path_logs + '/config.json')


def mat2str(matrix):
    return np.array2string(matrix, separator=',', max_line_width=np.inf)


def set_config(model_1, model_2):
    path_model_1 = '/scratch/compmatsc/neumann/models/' + dir_name + '/' + model_1 + '/'
    path_model_2 = '/scratch/compmatsc/neumann/models/' + dir_name + '/' + model_2 + '/'

    config = AZh.Config(
        game="oware",  # <======   change game here
        MC_matches=False,
        path=path_logs,
        path_model_1=path_model_1,
        path_model_2=path_model_2,
        checkpoint_number_1=None,
        checkpoint_number_2=None,
        use_solver=False,
        use_two_solvers=False,
        solver_1_temp=None,
        solver_2_temp=None,
        logfile='matches',
        learning_rate=0,
        weight_decay=0,

        temperature=0.25,
        evaluators=80,
        uct_c=2,
        max_simulations=300,
        policy_alpha=0.5,  # was 0
        evaluation_games=10,
        evaluation_window=10,

        nn_model=None,
        nn_width=None,
        nn_depth=None,
        observation_shape=None,
        output_size=None,

        quiet=True,
    )
    return config


def main(unused_argv):
    n_sizes = 7
    n_copies = 1
    nets = []
    for i in range(n_sizes):
        for j in range(n_copies):
            nets.append('ur_' + str(i) + '_' + str(j))

    n = len(nets)
    matches = np.zeros([n, n])

    for pair in combinations(range(n), 2):  # Loop over pairs of nets
        net_1 = nets[pair[0]]
        net_2 = nets[pair[1]]
        config = set_config(net_1, net_2)
        AZh.run_matches(config)
        n_evaluators = config.evaluators
        score = 0
        for ev in range(n_evaluators):
            with open(config.path + 'log-' + config.logfile + '-' + str(ev) + '.txt') as f:
                lines = f.readlines()
            score += float(lines[-2][-7:-2])
            # Note that the logs only contain scores up to 3 places after the decimal point.
            # Might be worth to check out later if that can be changed, but in any case
            # fluctuations have always been >0.001, so it's not a big deal.
        score = score / n_evaluators
        matches[pair] += (float(score) + 1) / 2


    # Calculate mean matrix (averaging over copies with same sizes):
    m = int(len(matches[0, :]) / n_copies)
    mean_mat = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            x = i * n_copies
            y = j * n_copies
            mat = matches[x:x + n_copies, y:y + n_copies]
            if i == j:
                continue
            mean_mat[i, j] = mat.mean()

    matches = matches + np.tril(np.ones([n, n]) - matches.transpose(), -1)
    print(matches)
    mean_mat = mean_mat + np.tril(np.ones([m, m]) - mean_mat.transpose(), -1)
    print(mean_mat)

    # Save matrix to file. Not as text since it's too big.
    with open(path_logs + "/matrix.npy", 'wb') as f:
        np.save(f, matches)
    with open(path_logs + "/mean_matrix.txt", "w") as f:
        f.write(mat2str(mean_mat))


if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)