import numpy as np
import pickle
from src.data_analysis.state_value.solver_values import solver_optimal_moves
from src.general.general_utils import models_path, game_path
from src.data_analysis.state_value.value_prediction import get_model_policy_estimator
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots

"""
Plot the effect of temperature on agent's strategy quality.
Calculates the probability that an agent takes an optimal move.
Using the same states from plot_solver_value_loss.py
(run plot_solver_value_loss.py first)
"""

plt.style.use(['science','nature'])
TEMPERATURES = [0.01, 0.02, 0.04, 0.07, 0.1 , 0.14, 0.2 , 0.25, 0.32, 0.45, 0.6, 0.8 , 1, 1.4 , 2, 3, 5]
ESTIMATORS = [0, 1, 2, 3, 4, 5, 6]
N_SAMPLES = 200


def save_optimal_moves() -> None:
    print('Calculating optimal policies.')
    with open('../plot_data/solver/state_counter.pkl', 'rb') as f:
        state_counter = pickle.load(f)
    chunk_size = 100
    optimal_moves = dict()
    all_keys = [key for key,_ in state_counter.frequencies.most_common()]
    all_keys = all_keys[:N_SAMPLES*100]
    for i in tqdm(range(0, len(state_counter.frequencies), chunk_size), desc="Estimating optimal moves"):
        keys = all_keys[i:i+chunk_size]
        vals = solver_optimal_moves([state_counter.serials[key] for key in keys])
        optimal_moves.update({key: val for key, val in zip(keys, vals)})
    with open('../plot_data/solver/optimal_moves.pkl', 'wb') as f:
        pickle.dump(optimal_moves, f)


def _get_optimal_policies(path_model: str, serials: list[str], temperature: float) -> list:
    model_policies = get_model_policy_estimator('connect_four', path_model)
    return model_policies(serials, temperature=temperature)


def save_agent_probs():
    print('Calculating agent probabilities')
    with open('../plot_data/solver/state_counter.pkl', "rb") as f:
        state_counter = pickle.load(f)
    with open('../plot_data/solver/optimal_moves.pkl', "rb") as f:
        optimal_moves = pickle.load(f)
    n_copies = 1
    path = models_path() + game_path('connect_four')
    keys = [key for key,_ in state_counter.frequencies.most_common()]
    for i,key in enumerate(keys):
        if all(optimal_moves[key]):
            keys.pop(i)
    keys = keys[:N_SAMPLES]
    serials = [state_counter.serials[key] for key in keys]
    prob_of_optimal_move = {est: {t: [] for t in TEMPERATURES} for est in ESTIMATORS}
    for est in ESTIMATORS:
        for copy in range(n_copies):
            model_name = f'q_{est}_{copy}'
            print(model_name)
            path_model = path + model_name + '/'
            for t in TEMPERATURES:
                print('Temperature:', t)
                policies = _get_optimal_policies(path_model, serials, t)
                for i,policy in enumerate(policies):
                    prob_optimal = np.dot(policy, optimal_moves[keys[i]])
                    prob_of_optimal_move[est][t].append(prob_optimal)
    with open('../plot_data/solver/temp_probabilities.pkl', 'wb') as f:
        pickle.dump(prob_of_optimal_move, f)


def plot_policy_degradation(load_data: bool = True, res: int = 300):
    print('~~~~~~~~~~~~~~~~~~~ Plotting policy degradation with temperature (appendix) ~~~~~~~~~~~~~~~~~~~')
    if not load_data:
        save_optimal_moves()
        save_agent_probs()
    with open('../plot_data/solver/temp_probabilities.pkl', 'rb') as f:
        prob_of_optimal_move = pickle.load(f)

    tf =16
    fig = plt.figure(figsize=(12, 6))

    par: np.ndarray = np.load('src/config/parameter_counts/connect_four.npy')
    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())
    norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    cbar = plt.colorbar(sm)
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=tf)

    plt.xlabel('Temperature', fontsize=tf)
    plt.ylabel('Probability to play an optimal move', fontsize=tf)
    for est in ESTIMATORS:
        y = [np.mean(prob_of_optimal_move[est][t]) for t in TEMPERATURES]
        err = [np.std(prob_of_optimal_move[est][t])/np.sqrt(N_SAMPLES) for t in TEMPERATURES] # SEM
        plt.errorbar(TEMPERATURES, y, yerr=[err, err], fmt='-o', color=cm.viridis(color_nums[est]), linewidth=2, markersize=8)
    plt.xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=tf-2)

    plt.axvline(x=0.45, color='black', linestyle='--', label='Fig. 3 B cutoff', linewidth=2)
    plt.text(0.4, 0.1, 'Policy quality\n unaffected', transform=plt.gca().transAxes, fontsize=tf, weight='bold')
    plt.text(0.65, 0.1, 'Policy quality\n decreasing', transform=plt.gca().transAxes, fontsize=tf, weight='bold')
    plt.legend(fontsize=tf)
    plt.title('Decrease of policy quality with temperature', fontsize=tf+4, loc='left')
    fig.tight_layout()
    fig.savefig('./plots/policy_degradation.png', dpi=res)
