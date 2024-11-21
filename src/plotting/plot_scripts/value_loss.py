import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import numpy as np
import scienceplots
from src.plotting.plot_utils import aligned_title, gaussian_average
from src.data_analysis.state_value.alpha_beta_pruning_time import save_pruning_time
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_value.value_loss import value_loss
from src.general.general_utils import models_path, game_path

plt.style.use(['science','nature','grid'])


def _generate_loss_curves(env, data_labels, n_copies):
    print('Generating loss curves for ', env)
    path = models_path()
    state_counter = StateCounter(env, save_serial=True, save_value=True)
    loss_curves = dict()
    for label in data_labels:
        for copy in range(n_copies):
            model_name = f'q_{label}_{copy}'
            print(model_name)
            model_path = path + game_path(env) + model_name + '/'
            state_counter.reset_counters()
            state_counter.collect_data(path=model_path, max_file_num=39)
            state_counter.normalize_counters()
            state_counter.prune_low_frequencies(10)
            loss = value_loss(env, model_path, state_counter=state_counter)
            loss_curves[model_name] = loss
    with open('../plot_data/value_loss/training_loss/loss_curves_'+env+'.pkl', 'wb') as f:
        pickle.dump(loss_curves, f)


def _generate_gaussian_smoothed_loss(labels: str, loss_curves, sigma: float):
    for label in tqdm(labels):
        curves = [np.array(loss_curves[f'q_{label}_{copy}']) for copy in range(6)]
        l = min([len(curve) for curve in curves])
        curves = [curve[:l] for curve in curves]
        y = np.mean(curves, axis=0)
        y = gaussian_average(y, sigma=sigma, cut_tail=True)
        with open('../plot_data/value_loss/training_loss/gaussian_loss_connect_four_'+str(label)+'.pkl', 'wb') as f:
            pickle.dump(y, f)


def _generate_solver_gaussian_loss(losses, label: int, l_max, sigma: float):
    y = losses[label]
    y = y[:l_max]
    y = gaussian_average(y, sigma=sigma, cut_tail=True)
    with open('../plot_data/solver/gaussian_loss'+str(label)+'.pkl', 'wb') as f:
        pickle.dump(y, f)


def connect4_loss_plots(load_data=True, res=300):
    print('Plotting Connect Four value loss plots')
    tf =12
    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    par = np.load('src/config/parameter_counts/connect_four.npy')
    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

    titles = [r'$\bf{A.}$ Value loss (train set)',
              r'$\bf{B.}$ Value loss (ground truth)',
              r'$\bf{C.}$ Time required, $\alpha$-$\beta$ pruning']
    sigma = 0.15
    labels = [0, 1, 2, 3, 4, 5, 6]
    l_max = 0
    for i, ax in enumerate(axs):
        if i == 0:
            print('[1/3] Plotting training loss')
            if not load_data:
                _generate_loss_curves('connect_four', labels, 6)
            with open('../plot_data/value_loss/training_loss/loss_curves_connect_four.pkl', 'rb') as f:
                loss_curves = pickle.load(f)
            if not load_data:
                _generate_gaussian_smoothed_loss(labels=labels, loss_curves=loss_curves, sigma=sigma)
            for label in tqdm([0, 1, 2, 3, 4, 5, 6]):
                with open('../plot_data/value_loss/training_loss/gaussian_loss_connect_four_'+str(label)+'.pkl', 'rb') as f:
                    y = pickle.load(f)
                l_max = max(l_max, len(y))
                ax.plot(np.arange(len(y))+1, y, color=cm.viridis(color_nums[label]))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=tf-2)
            ax.set_ylabel('Loss',fontsize=tf)
            del loss_curves
        if i == 1:
            print('[2/3] Plotting ground-truth loss')
            print('x axis length:', l_max)
            with open('../plot_data/solver/loss_curves.pkl', "rb") as f:
                losses = pickle.load(f)
            for label in tqdm([0, 1, 2, 3, 4, 5, 6]):
                if not load_data:
                    _generate_solver_gaussian_loss(losses, label, l_max, sigma)
                with open('../plot_data/solver/gaussian_loss'+str(label)+'.pkl', 'rb') as f:
                    y = pickle.load(f)
                ax.plot(np.arange(len(y))+1, y, color=cm.viridis(color_nums[label]))
            ax.set_xscale('log')
            ax.set_yscale('linear')
            ax.tick_params(axis='both', which='major', labelsize=tf-2)
            ax.set_ylabel('Loss',fontsize=tf)
        if i == 2:
            print('[3/3] Plotting AB pruning complexity')
            if not load_data:
                save_pruning_time(generate_counter=True, plot=False)
            with open('../plot_data/ab_pruning/data.pkl', 'rb') as f:
                ab_data = pickle.load(f)
            x = ab_data['x']
            y = np.array(ab_data['g_mean'])
            gstd = np.array(ab_data['gstd'])
            err = np.array([y*(1-1/gstd), y*(gstd-1)])
            c = -15 # Cut off hardware-limit plateau
            ax.errorbar(x[:c], y[:c], yerr=err[:,:c], fmt='-o')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=tf-2)
            ax.set_ylabel('CPU time (s)',fontsize=tf)
        ax.set_xlabel('State rank',fontsize=tf)
        aligned_title(ax, title=titles[i],font=tf+4)
    
    # Colorbar:
    norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    cbar = fig.colorbar(sm, ax=axs[2])
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=tf)

    fig.tight_layout()
    fig.savefig('./plots/connect4_value_loss.png', dpi=res)


def _state_loss(env, path, checkpoint_number=10_000):
    state_counter = StateCounter(env, save_serial=True, save_value=True, save_turn_num=True, cut_extensive=False)
    state_counter.collect_data(path=path, max_file_num=50)
    state_counter.normalize_counters()
    state_counter.prune_low_frequencies(threshold=10)
    turn_mask = state_counter.late_turn_mask(threshold=40)
    loss = value_loss(env, path, state_counter=state_counter, checkpoint_number=checkpoint_number, num_chunks=400)
    total_loss = 0
    counts = 0
    i=0
    for _, count in state_counter.frequencies.most_common():
        total_loss += loss[i]*count
        i+=1
        counts += count
    print('Model loss on train set:', total_loss/counts)
    return loss, turn_mask


def _gereate_oware_loss_curves(data_labels: list[int], n_copies: int):
    print('Generating loss curves for Oware')
    env='oware'
    loss_types = ('later_turns','early_turns','every_state')
    losses = {label: {k: None for k in loss_types} for label in data_labels}
    for copy in range(n_copies):
        for label in data_labels:
            model_name = f'q_{label}_{copy}'#
            print(model_name)
            model_path = models_path() + game_path(env) + model_name + '/'
            loss, turn_mask = _state_loss(env, model_path)
            losses[label]['later_turns'] = loss*turn_mask
            losses[label]['early_turns'] = loss*(~turn_mask)
            losses[label]['every_state'] = loss
        with open(f'../plot_data/value_loss/late_turns/loss_curves_{env}_{copy}.pkl', 'wb') as f:
            pickle.dump(losses, f)


def _gereate_oware_checkpoint_loss_curves(label: int, n_copies: int, checkpoints: list[int]):
    print('Generating loss curves for Oware checkpoints')
    env='oware'
    loss_types = ('later_turns','early_turns','every_state')
    losses = {checkpoint: {k: None for k in loss_types} for checkpoint in checkpoints}
    for copy in range(n_copies):
        for checkpoint in checkpoints:
            model_name = f'q_{label}_{copy}'#
            print(model_name)
            model_path = models_path() + game_path(env) + model_name + '/'
            loss, turn_mask = _state_loss(env, model_path, checkpoint_number=checkpoint)
            losses[checkpoint]['later_turns'] = loss*turn_mask
            losses[checkpoint]['early_turns'] = loss*(~turn_mask)
            losses[checkpoint]['every_state'] = loss
        with open(f'../plot_data/value_loss/late_turns/checkpoint_loss_curves_{env}_{copy}.pkl', 'wb') as f:
            pickle.dump(losses, f)


def _oware_gaussian_smoothed_loss(labels: list[int], n_copies: int, sigma: float, use_checkpoints: bool = False):
    print('Smoothing Oware loss curves')
    loss_types = ('later_turns','early_turns','every_state')
    averaged_curves = {label: {k: None for k in ('later_turns','early_turns','every_state')} for label in labels}
    for copy in range(n_copies):
        if use_checkpoints:
            with open(f'../plot_data/value_loss/late_turns/checkpoint_loss_curves_oware_{copy}.pkl', "rb") as f:
                loss_curves = pickle.load(f)
        else:
            with open(f'../plot_data/value_loss/late_turns/loss_curves_oware_{copy}.pkl', "rb") as f:
                loss_curves = pickle.load(f)
        for label in labels:
            for t in loss_types:
                all_curves = averaged_curves[label][t]
                new_curve = loss_curves[label][t]
                if all_curves is None:
                    averaged_curves[label][t] = new_curve
                else:
                    l = min(len(all_curves), len(new_curve))
                    averaged_curves[label][t] = all_curves[:l] * new_curve[:l]
    for label in labels:
            for t in loss_types:
                averaged_curves[label][t] = (averaged_curves[label][t])**(1/n_copies)

    loss_types = ('later_turns','early_turns','every_state')
    smooth_losses = {label: {k: None for k in loss_types} for label in labels}
    ranks = {label: {k: None for k in loss_types} for label in labels}
    for t in loss_types:
        for label in tqdm(labels):
            curve = np.array(averaged_curves[label][t])
            mask = curve > 0
            smooth_losses[label][t], ranks[label][t] = gaussian_average(curve, sigma=sigma, cut_tail=True, mask=mask)
    filename = 'checkpoint_loss_oware_total.pkl' if use_checkpoints else 'loss_oware_total.pkl'
    with open(f'../plot_data/value_loss/late_turns/gaussian_'+filename, 'wb') as f:
        pickle.dump([smooth_losses, ranks], f)


def oware_value_loss(load_data=True, res=300):
    print('~~~~~~~~~~~~~~~~~~~ Plotting oware value loss ~~~~~~~~~~~~~~~~~~~')
    tf =12
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.))
    sigma = 0.15
    labels = [0, 1, 2, 3, 4, 5, 6]
    n_copies = 1
    if not load_data:
        _gereate_oware_loss_curves(labels, n_copies)
        _oware_gaussian_smoothed_loss(labels, n_copies, sigma)

    with open('../plot_data/value_loss/late_turns/gaussian_loss_oware_total.pkl', "rb") as f:
        losses, ranks =  pickle.load(f)
    par = np.load('src/config/parameter_counts/oware.npy')
    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

    loss_types = ('every_state', 'early_turns', 'later_turns')
    titles = [r'$\bf{A.}$ Oware value loss', r'$\bf{B.}$ Early-turn loss', r'$\bf{C.}$ Late-turn loss']
    ylim = None
    for i,ax in enumerate(axes.flat):
        t = loss_types[i]
        aligned_title(ax, title=titles[i],font=tf+4)
        for label in labels:
            x = ranks[label][t]
            y = losses[label][t]
            ax.plot(x, y, color=matplotlib.cm.viridis(color_nums[label]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=tf-2)
        ax.set_xlabel('State rank',fontsize=tf)
        if i == 0:
            ax.set_ylabel('Loss',fontsize=tf)
            ax.set_ylim(bottom=0.8*10**-3)
            ylim = ax.get_ylim()
            late_start = min(ranks[label]['later_turns'][0] for label in labels)
            ax.axvline(x=late_start, linestyle='--', color='black', label='First late-turn states')
        else:
            ax.set_ylim(ylim)
        if i==2:
            ax.set_xlim(left=10**0)
            # Add zoom-in inset
            axin = ax.inset_axes([0.02, 0.02, 0.96, 0.48])
            for label in labels:
                x = ranks[label][t]
                y = losses[label][t]
                axin.plot(x, y, color=matplotlib.cm.viridis(color_nums[label]))
            axin.set_xscale('log')
            axin.set_yscale('log')
            axin.set_ylim(bottom=6*10**-2, top=2.8*10**-1)
            axin.set_xlim(left=10**2, right=0.8*10**5)
            axin.tick_params(axis='both', which='both', labelsize=0)
            ax.indicate_inset_zoom(axin, edgecolor="black",linewidth=2)

    norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
    # create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    cbar = fig.colorbar(sm, ax=axes[2]) # attach to plot 2, rather than to inset
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=tf)

    fig.tight_layout()
    fig.savefig('./plots/oware_value_loss.png', dpi=res)


def oware_checkpoint_value_loss(load_data=True, res=300):
    print('~~~~~~~~~~~~~~~~~~~ Plotting oware value loss (over checkpoints) ~~~~~~~~~~~~~~~~~~~')
    tf =12
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 2.5))
    sigma = 0.15
    label = 6
    checkpoints = [20, 30, 50, 70, 100, 150, 230, 340, 510, 770, 1150, 1730, 2590, 3880, 5820, 8730, 10000]
    used_checkpoints = [50,  150,  510, 1150,  2590,  5820, 10000]
    n_copies = 1
    if not load_data:
        _gereate_oware_checkpoint_loss_curves(label, n_copies, checkpoints)
        _oware_gaussian_smoothed_loss(checkpoints, n_copies, sigma, use_checkpoints=True)

    with open('../plot_data/value_loss/late_turns/gaussian_checkpoint_loss_oware_total.pkl', "rb") as f:
        losses, ranks =  pickle.load(f)
    log_par = np.log(checkpoints)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

    loss_types = ('every_state', 'early_turns', 'later_turns')
    titles = [r'$\bf{A.}$ Large agent value loss', r'$\bf{B.}$ Early-turn loss', r'$\bf{C.}$ Late-turn loss']
    ylim = None
    for i,ax in enumerate(axes.flat):
        t = loss_types[i]
        aligned_title(ax, title=titles[i],font=tf+4)
        for i, label in enumerate(checkpoints):
            if label not in used_checkpoints:
                continue
            x = ranks[label][t]
            y = losses[label][t]
            ax.plot(x, y, color=matplotlib.cm.cividis(color_nums[i]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=tf-2)
        ax.set_xlabel('State rank',fontsize=tf)
        ax.set_ylabel('Loss',fontsize=tf)
        if i == 0:
            ax.set_ylabel('Loss',fontsize=tf)
            ax.set_ylim(bottom=0.8*10**-3)
            ylim = ax.get_ylim()
            late_start = min(ranks[label]['later_turns'][0] for label in checkpoints)
            ax.axvline(x=late_start, linestyle='--', color='black', label='First late-turn states')
        else:
            ax.set_ylim(ylim)
        if i==2:
            ax.set_xlim(left=10**0)
            # Add zoom-in inset
            for label in checkpoints:
                x = ranks[label][t]
                y = losses[label][t]
                ax.plot(x, y, color=matplotlib.cm.cividis(color_nums[label]))
            ax.set_ylim(bottom=6*10**-2, top=2.8*10**-1)
            ax.set_xlim(left=10**2, right=0.8*10**5)
    norm = matplotlib.colors.LogNorm(vmin=min(checkpoints)/max(checkpoints), vmax=1)
    # create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('cividis'), norm=norm)
    cbar = fig.colorbar(sm, ax=axes[2]) # attach to plot 2, rather than to inset
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Training time (normalized)', rotation=90, fontsize=tf)

    fig.tight_layout()
    fig.savefig('./plots/oware_value_loss_checkpoints.png', dpi=res)
