from absl import app
import os
from shutil import copyfile
from open_spiel.python.utils import spawn
import sys
from itertools import combinations
from src.alphazero_scaling import AZ_helper_lib as AZh


"""
Run games at the specified temepratures.
Generates a big bulk of games for calculating board state distribution later.
"""

# Name of the directory where the models are saved. Don't forget to change 'game' in config.
dir_name = 'connect_four'

checkpoint_number = 10000
temps = [0.07, 0.1, 0.14, 0.2, 0.25, 0.32, 0.45, 0.6, 0.8, 1, 1.4, 2, 3, 5, 0.04, 0.02, 0.01]
i = int(sys.argv[1])
temperature = temps[i]
# Save all logs to:
path_logs = './models/matches/temperature_matches/'+str(dir_name)+'/temp_num_' +str(i) + '/'
if not os.path.exists(path_logs):
    os.makedirs(path_logs)


def set_config(model_1, model_2, n_games, output_dir):
    path_model_1 = './models/' + dir_name + '/' + model_1 + '/'
    path_model_2 = './models/' + dir_name + '/' + model_2 + '/'
    # Create the directory, then copy the config.json file into it: (otherwise crashes)
    output_path = path_logs + output_dir + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        copyfile('./matches/config.json', output_path + '/config.json')
    config = AZh.Config(
        game="connect_four",  # <======   change game here
        MC_matches=False,
        path=output_path,
        path_model_1=path_model_1,
        path_model_2=path_model_2,
        checkpoint_number_1=checkpoint_number,
        checkpoint_number_2=checkpoint_number,
        use_solver=False,
        logfile='matches',
        learning_rate=0,
        weight_decay=0,
        use_two_solvers=False,
        solver_1_temp=None,
        solver_2_temp=None,

        temperature=temperature,
        evaluators=80,
        uct_c=2,
        max_simulations=300,  # 300
        policy_alpha=0.5,  # was 0
        evaluation_games=n_games,
        evaluation_window=n_games,
        
        nn_model=None,
        nn_width=None,
        nn_depth=None,
        observation_shape=None,
        output_size=None,

        quiet=True,
    )
    return config


def main(unused_argv):
    max_q = 6
    n_copies = 3 #6
    nets = []
    # Enumerate all models. The order is: First by size, then by copy number.
    for i in range(max_q + 1):
        for j in range(n_copies):
            nets.append('q_' + str(i) + '_' + str(j))
    n = len(nets)
    counter = 0
    total = n*(n-1)/2
    n_games = 186  # ~ 1500/80
    for pair in combinations(range(n), 2):  # Loop over pairs of nets
        counter += 1
        print('Percent complete: %.0f%%' % (counter*100/total))
        net_1 = nets[pair[0]]
        net_2 = nets[pair[1]]
        config = set_config(net_1, net_2, n_games=n_games, output_dir=net_1 + '_vs_' + net_2)
        AZh.run_matches(config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
                                        