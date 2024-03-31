import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
import scienceplots
from late_turn_loss import smooth

plt.style.use(['science','nature','grid'])

def game_turns():
    tf =12
    # Create figure and subplots
    fig = plt.figure(figsize=(12, 6))

    # Define grid for subplots
    # Divide the figure into 3 columns and 2 rows
    # The 5th subplot (index 4) spans 2 rows and 1 column
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax5 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

    # Plots 1-4
    square_plots = [ax1, ax2, ax3, ax4]
    colors = ['blue', 'purple', 'green', 'olive']
    envs = ['connect_four', 'pentago', 'oware', 'checkers']
    env_names =['Connect Four', 'Pentago', 'Oware', 'Checkers']
    for i, ax in enumerate(tqdm(square_plots, desc='Raw data plots')):
        with open('../plot_data/turns/raw_turns_'+envs[i]+'.pkl', "rb") as f:
            y =  pickle.load(f)
        x = np.arange(len(y)) + 1
        ax.scatter(x, y, color=colors[i], s=40 * 3 / (10 + x), label=env_names[i])
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Specific legend positions
        if i<2:
            ax.legend(loc='upper left', fontsize=tf)
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=tf)
        ax.tick_params(axis='both', which='major', labelsize=tf-2)


    # Add axis labels to each subplot
    ax1.set_ylabel('Turn number',fontsize=tf)
    ax3.set_xlabel('State rank',fontsize=tf)
    ax3.set_ylabel('Turn number',fontsize=tf)
    ax4.set_xlabel('State rank',fontsize=tf)

    # PLot num. 5
    for i, env in enumerate(tqdm(envs, desc='Turn ratio plot')):
        with open('../plot_data/turns/turn_ratio_'+env+'.pkl', "rb") as f:
            bin_x, ratio =  pickle.load(f)
        ax5.plot(bin_x, ratio, color=colors[i], label=env_names[i], linewidth=3.0)
    ax5.set_xlabel('State rank',fontsize=tf+2)
    ax5.set_ylabel('Ratio of late ($>40$) turns',fontsize=tf+2)
    ax5.set_xscale('log')
    ax5.set_ylim(top=1)
    ax5.legend(loc="upper left", framealpha=0.6, fontsize=tf)
    ax5.tick_params(axis='both', which='major', labelsize=tf)

    # Set titles for each subplot
    def aligned_title(ax, title):
        bbox = ax.get_yticklabels()[-1].get_window_extent()
        x,_ = ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
        ax.set_title(title, ha='left',x=x,fontsize=tf+4)
    aligned_title(ax1, r'$\bf{a.}$ Turn distribution')
    aligned_title(ax5, r'$\bf{b.}$ Turn ratios')

    plt.tight_layout()
    fig.savefig('./plots/turns.png', dpi=900)

def smooth(vec):
    """return a smoothed vec with values averaged with their neighbors."""
    a = 0.5
    filter = np.array([a, 1, a]) / (1 + 2 * a)
    new_vec = np.convolve(vec, filter, mode='same')
    new_vec[0] = (vec[0] + a * vec[1]) / (1 + a)
    new_vec[-1] = (vec[-1] + a * vec[-2]) / (1 + a)
    return new_vec

def oware_value_loss():
    par = np.load('src/config/parameter_counts/oware.npy')
    tf =12
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    with open('../plot_data/value_loss/loss_curves_oware.pkl', "rb") as f:
        loss_values, rank_values =  pickle.load(f)

    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())

    loss_types = ('every_state', 'early_turns', 'later_turns')
    data_labels = [0, 1, 2, 3, 4, 5, 6] 
    titles = ['Oware value loss', 'Loss on early turns', 'Loss on late turns']
    for i,ax in enumerate(axes.flat):
        t = loss_types[i]
        ax.set_title(titles[i], fontsize=tf+4)
        #figure.preamble()
        for label in data_labels:
            x = rank_values[label][t]
            y = loss_values[label][t]
            y = smooth(y)
            plt.plot(x, y, color=matplotlib.cm.viridis(color_nums[label]))

        plt.xscale('log')
        plt.yscale('log')

    norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
    # create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    cbar = fig.colorbar(sm)
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=tf)

    plt.tight_layout()
    fig.savefig('./plots/oware_value_loss.png', dpi=900)

#game_turns()
oware_value_loss()


