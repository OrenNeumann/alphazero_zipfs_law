import numpy as np
import pickle
from src.data_analysis.state_value.solver_values import solver_optimal_moves
from src.general.general_utils import models_path, game_path
from src.data_analysis.state_value.value_prediction import get_model_policy_estimator
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Plot the effect of temperature on agent's strategy quality.
Calculates the probability that an agent takes an optimal move.
Currently using the same states from plot_solver_value_loss.py
(run plot_solver_value_loss.py first)
"""

def save_optimal_moves():
    with open('../plot_data/solver/state_counter.pkl', 'rb') as f:
        state_counter = pickle.load(f)
    chunk_size = 100
    optimal_moves = dict()
    for i in tqdm(range(0, len(state_counter.frequencies), chunk_size), desc="Estimating optimal moves"): 
        keys = list(state_counter.frequencies.keys())[i:i+chunk_size]
        vals = solver_optimal_moves([state_counter.serials[key] for key in keys])
        optimal_moves.update({key: val for key, val in zip(keys, vals)})
    with open('../plot_data/solver/optimal_moves.pkl', 'wb') as f:
        pickle.dump(optimal_moves, f)

def _get_optimal_policies(path_model, serials, temperature):
    model_policies = get_model_policy_estimator('connect_four', path_model)
    return model_policies(serials, temperature=temperature)

def plot_policy_degradation():
    with open('../plot_data/solver/state_counter.pkl', "rb") as f:
        state_counter = pickle.load(f)
    with open('../plot_data/solver/optimal_moves.pkl', "rb") as f:
        optimal_moves = pickle.load(f)
    print('Plotting policy degradation')
    temps = [0.01, 0.02, 0.04, 0.07, 0.1 , 0.14, 0.2 , 0.25, 0.32, 0.45, 0.6, 0.8 , 1, 1.4 , 2, 3, 5]
    estimators = [0, 1, 2, 3, 4, 5, 6]
    n_copies = 1
    n_samples = 10
    path = models_path() + game_path('connect_four')
    keys = [key for key,_ in state_counter.frequencies.most_common()]
    for i,key in enumerate(keys):
        if all(optimal_moves[key]):
            keys.pop(i)
    #keys = np.random.choice(keys,n_samples,replace=False)
    keys = keys[:n_samples]
    print(keys)
    serials = [state_counter.serials[key] for key in keys]
    prob_of_optimal_move = {est: {t: [] for t in temps} for est in estimators}
    for est in estimators:
        for copy in range(n_copies):
            model_name = f'q_{est}_{copy}'
            print(model_name)
            path_model = path + model_name + '/'
            for t in temps:
                print('Temperature:', t)
                policies = _get_optimal_policies(path_model, serials, t)
                for i,policy in enumerate(policies):
                    prob_optimal = np.dot(policy, optimal_moves[keys[i]])
                    prob_of_optimal_move[est][t].append(prob_optimal)
                    print(policy)
                    print(optimal_moves[key])
    
    with open('../plot_data/solver/temp_probabilities.pkl', 'wb') as f:
        pickle.dump(prob_of_optimal_move, f)

    par = np.load('src/config/parameter_counts/connect_four.npy')
    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())
    norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
    # create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    cbar = plt.colorbar(sm)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=16)

    plt.xlabel('Temperature')
    plt.ylabel('Probability of optimal move')
    for est in estimators:
        y = [np.mean(prob_of_optimal_move[est][t]) for t in temps]
        err = [np.std(prob_of_optimal_move[est][t])/np.sqrt(n_samples) for t in temps] # SEM
        plt.errorbar(temps, y, yerr=[err, err], fmt='-o', color=cm.viridis(color_nums[est]))
    plt.xscale('log')
    plt.axvline(x=0.45, color='black', linestyle='--', label='data cutoff')
    plt.title('Decrease of policy quality with temperature')
    plt.savefig('plots/policy_degradation.png')

plot_policy_degradation()
