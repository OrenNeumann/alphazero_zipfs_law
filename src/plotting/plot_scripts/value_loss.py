import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
import scienceplots
from src.plotting.plot_utils import aligned_title

plt.style.use(['science','nature','grid'])

def connect4_loss_plots():
    print('Plotting Connect Four value loss plots')
    tf =12
    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    with open('../plot_data/ab_pruning/data.pkl', 'rb') as f:
        ab_data = pickle.load(f)

    titles = [r'$\bf{a.}$ Value loss on train set',
              r'$\bf{b.}$ Value loss on ground truth',
              r'$\bf{c.}$ Alpha-beta-pruning complexity']
    for i, ax in enumerate(axs):
        x = ab_data['x']
        y = np.array(ab_data['g_mean'])#
        gstd = np.array(ab_data['gstd'])
        err = np.array([y*(1-1/gstd), y*(gstd-1)])
        c = -15 # Cut off hardware-limit plateau
        ax.errorbar(x[:c], y[:c], yerr=err[:,:c], fmt='-o')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=tf-2)
        if i==0:
            ax.set_ylabel('Loss',fontsize=tf)
        if i==2:
            ax.set_ylabel('CPU time (s)',fontsize=tf)
        ax.set_xlabel('State rank',fontsize=tf)
        aligned_title(ax, title=titles[i],font=tf+4)

    plt.tight_layout()
    fig.savefig('./plots/connect4_value_loss.png', dpi=900)
