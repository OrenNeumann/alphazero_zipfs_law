import numpy as np
import pickle
from src.data_analysis.state_value.solver_values import solver_values
from src.general.general_utils import models_path, game_path
from src.data_analysis.gather_agent_data import gather_data
from src.data_analysis.state_value.value_loss import value_loss
from src.plotting.plot_utils import BarFigure, incremental_bin, smooth
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
Plot connect four loss with solver values as ground truth.
"""

def save_solver_values(data_labels: list[int], file_num: int = 1):
    env = 'connect_four'
    state_counter = gather_data(env, data_labels, max_file_num=file_num, save_serial=True)
    state_counter.prune_low_frequencies(10)
    chunk_size = 1000
    true_values = dict()
    for i in tqdm(range(0, len(state_counter.frequencies), chunk_size), desc="Estimating solver values"): 
        keys = list(state_counter.frequencies.keys())[i:i+chunk_size]
        solver_values([state_counter.serials[key] for key in keys])
        true_values.update({key: val for key, val in zip(keys, solver_values)})
    with open('../plot_data/solver/solver_values.pkl', 'wb') as f:
        pickle.dump({'state_counter': state_counter,
                     'true_values': true_values}, f)


def plot_solver_value_loss():
    with open('../plot_data/solver/solver_values.pkl', "rb") as f:
        vars = pickle.load(f)
    state_counter, true_values = vars['state_counter'], vars['solver_values']
    print('loss part...')
    estimators = [0, 1, 2, 3, 4, 5, 6]
    n_copies = 6
    path = models_path() + game_path('connect_four')
    losses = {est: np.zeros(len(state_counter.frequencies)) for est in estimators}
    for est in estimators:
        for copy in range(n_copies):
            model_name = f'q_{est}_{copy}'
            print(model_name)
            path_model = path + model_name + '/'
            loss = value_loss(env='connect_four', path_model=path_model, state_counter=state_counter, 
                              num_chunks=40, values=true_values)
            losses[est] += loss
        losses[est] /= n_copies
    loss_values, rank_values = bin_loss_curves(estimators, losses)

    par = np.load('src/config/parameter_counts/connect_four.npy')
    fig = BarFigure(par=par, x_label='State rank', y_label='Loss', title='Ground-truth value loss', 
                    text_font=16, number_font=14)
    color_nums = fig.colorbar_colors()

    fig.preamble()
    for est in tqdm(estimators, desc='Plotting loss'):
        x = rank_values[est]
        y = loss_values[est]
        plt.scatter(x, y, color=cm.viridis(color_nums[est]))
    plt.xscale('log')
    fig.epilogue()
    fig.save('solver_value_loss')

    fig.fig_num += 1
    fig.preamble()
    for est in tqdm(estimators, desc='Plotting loss (smoothed)'):
        x = rank_values[est]
        y = loss_values[est]
        plt.scatter(x, smooth(y), color=cm.viridis(color_nums[est]))
    plt.xscale('log')
    fig.epilogue()
    fig.save('solver_value_loss_smoothed')


def bin_loss_curves(estimators, losses):
    """ Collect loss values in bins."""
    bins = incremental_bin(10**10)
    widths = (bins[1:] - bins[:-1])
    x = bins[:-1] + widths/2
    loss_values = {label: None for label in estimators}
    rank_values = {label: None for label in estimators}
    for est in estimators:
        loss = losses[est]
        ranks = np.arange(len(loss)) + 1
        # Calculate histogram.
        # np.histogram counts how many elements of 'ranks' fall in each bin.
        bin_count = np.histogram(ranks, bins=bins)[0]
        loss_sums = np.histogram(ranks, bins=bins, weights=loss)[0]
        # Divide sum to get average:
        mask = np.nonzero(bin_count)
        loss_values[est] = loss_sums[mask] / bin_count[mask]
        rank_values[est] = x[mask]
    with open('../plot_data/solver/loss_curves.pkl', 'wb') as f:
        pickle.dump({'loss_values': loss_values,
                     'rank_values': rank_values}, f)
    return loss_values, rank_values


save_solver_values(data_labels=[0, 2, 4, 6], file_num=1)
plot_solver_value_loss()
