import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
import scienceplots

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
        if i==2:
            with open('../plot_data/turns/raw_turns_'+envs[0]+'.pkl', "rb") as f:
                y =  pickle.load(f)
        else:
            with open('../plot_data/turns/raw_turns_'+envs[i]+'.pkl', "rb") as f:
                y =  pickle.load(f)
        x = np.arange(len(y)) + 1
        ax.scatter(x, y, color=colors[i], s=40 * 3 / (10 + x), label=env_names[i])
        ax.set_xscale('log')
        ax.set_yscale('log')
        if i<2:
            location = 'upper left'
        else:
            location = 'center left'
        ax.legend(loc=location, fontsize=tf)
        ax.tick_params(axis='both', which='major', labelsize=tf-2)


    # Add axis labels to each subplot
    ax1.set_ylabel('Turn number',fontsize=tf)
    ax3.set_xlabel('State rank',fontsize=tf)
    ax3.set_ylabel('Turn number',fontsize=tf)
    ax4.set_xlabel('State rank',fontsize=tf)

    # PLot num. 5
    for i, env in enumerate(tqdm(envs, desc='Turn ratio plot')):
        if i==2:
            continue
        with open('../plot_data/turns/turn_ratio_'+env+'.pkl', "rb") as f:
            bin_x, ratio =  pickle.load(f)
        ax5.plot(bin_x, ratio, color=colors[i], label=env_names[i], linewidth=3.0)
    ax5.set_xlabel('State rank',fontsize=tf+2)
    ax5.set_ylabel('Late turn ratio ($>40$)',fontsize=tf+2)
    ax5.set_xscale('log')
    ax5.set_ylim(top=1)
    ax5.legend(loc="upper left", fontsize=tf)

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

game_turns()
