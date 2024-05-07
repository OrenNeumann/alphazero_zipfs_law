import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from tqdm import tqdm
import numpy as np
import scienceplots
from src.plotting.plot_utils import aligned_title, smooth, gaussian_average
from src.data_analysis.state_value.alpha_beta_pruning_time import save_pruning_time

plt.style.use(['science','nature','grid'])


def _generate_gaussian_smoothed_loss(label, loss_curves, sigma):
    curves = [np.array(loss_curves[f'q_{label}_{copy}']) for copy in range(6)]
    l = min([len(curve) for curve in curves])
    l_max = max(l, l_max)
    curves = [curve[:l] for curve in curves]
    y = np.mean(curves, axis=0)
    y = gaussian_average(y, sigma=sigma, cut_tail=True)
    with open('../plot_data/value_loss/training_loss/gaussian_loss_connect_four_'+str(label)+'.pkl', 'wb') as f:
        pickle.dump(y, f)


def _generate_solver_gaussian_loss(losses, label, l_max, sigma):
    y = losses[label]
    y = y[:l_max]
    y = gaussian_average(y, sigma=sigma, cut_tail=True)
    with open('../plot_data/solver/gaussian_loss'+str(label)+'.pkl', 'wb') as f:
        pickle.dump(y, f)


def connect4_loss_plots(load_data=True, res=300):
    print('Plotting Connect Four value loss plots')
    tf =12
    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    par = np.load('src/config/parameter_counts/connect_four.npy')
    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

    titles = [r'$\bf{A.}$ Value loss (train set)',
              r'$\bf{B.}$ Value loss (ground truth)',
              r'$\bf{C.}$ Time required, $\alpha$-$\beta$ pruning']
    sigma = 0.15
    l_max = 0
    for i, ax in enumerate(axs):
        if i == 0:
            print('[1/3] Plotting training loss')
            with open('../plot_data/value_loss/training_loss/loss_curves_connect_four.pkl', 'rb') as f:
                loss_curves = pickle.load(f)
            for label in tqdm([0, 1, 2, 3, 4, 5, 6]):
                if not load_data:
                    _generate_gaussian_smoothed_loss(label, loss_curves, sigma)
                with open('../plot_data/value_loss/training_loss/gaussian_loss_connect_four_'+str(label)+'.pkl', 'rb') as f:
                    y = pickle.load(f)
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
            y = np.array(ab_data['g_mean'])#
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
    #cbar = fig.colorbar(sm, ax=axs[2],location='left') # attach to last plot
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=tf)

    fig.tight_layout()
    fig.savefig('./plots/connect4_value_loss.png', dpi=res)


def oware_value_loss(res=300):
    # using data from late_turn_loss.py, oware games had pruning=10
    print('~~~~~~~~~~~~~~~~~~~ Plotting oware value loss ~~~~~~~~~~~~~~~~~~~')
    
    tf =12
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    with open('../plot_data/value_loss/late_turns/loss_curves_oware.pkl', "rb") as f:
        loss_values, rank_values =  pickle.load(f)
    par = np.load('src/config/parameter_counts/oware.npy')
    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

    loss_types = ('every_state', 'early_turns', 'later_turns')
    data_labels = [0, 1, 2, 3, 4, 5, 6] 
    titles = ['Oware value loss', 'Early-turn loss', 'Late-turn loss']
    ylim = None
    for i,ax in enumerate(axes.flat):
        t = loss_types[i]
        aligned_title(ax, title=titles[i],font=tf+4)
        for label in data_labels:
            x = rank_values[label][t]
            y = loss_values[label][t]
            y = smooth(y)
            ax.plot(x, y, color=matplotlib.cm.viridis(color_nums[label]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=tf-2)
        ax.set_xlabel('State rank',fontsize=tf)
        if i == 0:
            ax.set_ylabel('Loss',fontsize=tf)
            ylim = ax.get_ylim()
            late_start = min(rank_values[label]['later_turns'][0] for label in data_labels)
            ax.axvline(x=late_start, linestyle='--', color='black', label='First late-turn states')
        else:
            ax.set_ylim(ylim)
        if i==1:
            #ax.set_ylabel(r'$\bf{=}$', rotation=0, fontsize=tf+6)
            pass
        if i==2:
            #ax.set_ylabel(r'$\bf{+}$', rotation=0, fontsize=tf+6)
            ax.set_xlim(left=10**0)
            # Add zoom-in inset
            axin = ax.inset_axes([0.02, 0.02, 0.96, 0.48])
            for label in data_labels:
                x = rank_values[label][t]
                y = loss_values[label][t]
                y = smooth(y)
                axin.plot(x, y, color=matplotlib.cm.viridis(color_nums[label]))
            axin.set_xscale('log')
            axin.set_yscale('log')
            axin.set_ylim(bottom=9*10**-2, top=2.8*10**-1)
            axin.set_xlim(left=10**2, right=2*10**5)
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


