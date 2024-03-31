import matplotlib.pyplot as plt
import pickle
import numpy as np
import tqdm

def game_turns():
    tf =10
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

    square_plots = [ax1, ax2, ax3, ax4]
    colors = ['blue', 'purple', 'green', 'olive']
    envs = ['connect_four', 'pentago', 'oware', 'checkers']
    env_names =['Connect Four', 'Pentago', 'Oware', 'Checkers']
    for i, ax in enumerate(tqdm(square_plots, desc='Raw data plots')):
        with open('../plot_data/turns/raw_turns_'+envs[0]+'.pkl', "rb") as f:
            y =  pickle.load(f)
        x = np.arange(len(y)) + 1
        ax.scatter(x, y, color=colors[i], s=40 * 3 / (10 + x), label=env_names[i])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc="upper left")
        ax.tick_params(axis='both', which='major', labelsize=tf)


    # Add axis labels to each subplot
    ax1.set_ylabel('Turn number',fontsize=tf)
    ax3.set_xlabel('State rank',fontsize=tf+2)
    ax3.set_ylabel('Turn number',fontsize=tf)
    ax4.set_xlabel('State rank',fontsize=tf+2)

    with open('../plot_data/turns/turn_ratio_'+envs[0]+'.pkl', "rb") as f:
        bin_x, ratio =  pickle.load(f)
    ax5.plot(bin_x, ratio, color=colors[0], label=env_names[0])
    ax5.set_xlabel('State rank',fontsize=tf+2)
    ax5.set_ylabel('Late turn ratio (>40)',fontsize=tf+2)
    ax5.set_xscale('log')
    ax5.legend(loc="upper left")

    ax5.tick_params(axis='both', which='major', labelsize=tf+2)

    # Set titles for each subplot
    bbox = ax1.get_yticklabels()[-1].get_window_extent()
    x,_ = ax1.transAxes.inverted().transform([bbox.x0, bbox.y0])
    ax1.set_title(r'$\bf{a.}$ Turn distribution', ha='left',x=x,fontsize=tf+2)
    bbox = ax5.get_yticklabels()[-1].get_window_extent()
    x,_ = ax5.transAxes.inverted().transform([bbox.x0, bbox.y0])
    ax5.set_title(r'$\bf{b.}$ Turn ratios', ha='left',x=x,fontsize=tf+2)

    plt.tight_layout()
    fig.savefig('./plots/turns.png', dpi=900)

game_turns()
