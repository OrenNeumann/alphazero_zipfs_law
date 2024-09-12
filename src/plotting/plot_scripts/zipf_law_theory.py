import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scienceplots
import pickle
from src.plotting.plot_utils import aligned_title
from src.data_analysis.state_frequency.state_counter import RandomGamesCounter

plt.style.use(['science','nature','grid'])

def _calculate_theory_distribution():
    print('Calculating Zipf law for toy-model game')
    depth = 8
    b_factor = 7
    l = 0
    for i in range(depth):
        l+= b_factor**(i+1)
    freq = np.zeros(l)
    ind = 0
    for d in range(depth):
        count = b_factor**(d+1)
        freq[ind:ind+count] = 1./count
        ind = ind+count
    # normalize:
    freq = freq/freq.min()
    with open('../plot_data/zipf_theory/theory_freq.pkl', 'wb') as f:
        pickle.dump(freq, f)


def _generate_random_games(env: str):
    print('Generating ' + env + ' random games')
    counter = RandomGamesCounter(env)
    counter.collect_data()
    with open('../plot_data/zipf_theory/random_'+env+'.pkl', 'wb') as f:
        pickle.dump(np.array([item[1] for item in counter.frequencies.most_common()]), f)


def plot_zipf_law_theory(load_data=True, res=300):
    """
    Plot Zipf laws of a theoretical toy model, and random rollouts.
    """
    tf =12
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    ax1 = axes[0]

    print('~~~~~~~~~~~~~~~~~~~ Plotting Zipf law theory ~~~~~~~~~~~~~~~~~~~')
    if not load_data:
        _calculate_theory_distribution()
    with open('../plot_data/zipf_theory/theory_freq.pkl', 'rb') as f:
        freq = pickle.load(f)
    x = np.arange(len(freq)) + 1

    # Fit:
    m =-1.
    c = np.log10(freq.max()) + 0.5
    exp = str(round(-m, 2))
    equation = r'$\alpha = ' + exp + '$'
    upper_bound = int(2.5*10**6)
    y_fit = 10 ** c * x[:upper_bound] ** m

    ax1.scatter(x,freq, color='dodgerblue', s=40 / (10 + np.log10(x)), alpha=1)
    ax1.plot(x[:upper_bound], y_fit, color='black', linewidth=1.3, alpha=0.7, label=equation)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('State rank',fontsize=tf)
    ax1.set_ylabel('Frequency',fontsize=tf)
    ax1.tick_params(axis='both', which='major', labelsize=tf-2)
    ax1.legend(fontsize=tf-2)
    aligned_title(ax1, title=r'$\bf{A.}$ Toy-model distribution',font=tf+4)

    ########################################################
    print('Plotting random games')
    ax2 = axes[1]
    envs = ['connect_four', 'pentago']
    labels = ['Connect Four', 'Pentago']
    colors = ['#377eb8', '#984ea3']
    if not load_data:
        for env in envs:
            _generate_random_games(env)
    for i, env in enumerate(envs):
        with open('../plot_data/zipf_theory/random_'+env+'.pkl', 'rb') as f:
            freq = pickle.load(f)
        x = np.arange(len(freq)) + 1
        if env == 'connect_four':
            ax2.scatter(x, freq, s=40 / (10 + np.log10(x)), color=colors[i], label=labels[i])
            low = 10**2
            up = int(len(freq)/10**2)
            x_nums = np.arange(up)[low:]
            [m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(freq[low:up]), deg=1, w=2 / x_nums)
            exp = str(round(-m, 2))
            equation = r'$\alpha = ' + exp + '$'
            y_fit = 10 ** c * x[:int(10**7)] ** m
            bound = np.argmax(y_fit < 1)
            ax2.plot(x[:bound], y_fit[:bound], color='black', linewidth=1.3, alpha=0.7, label=equation)
        else:
            ax2.scatter(x, freq, s=0.7*40 / (10 + np.sqrt(x)), color=colors[i], label=labels[i])

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('State rank',fontsize=tf)
    ax2.set_ylabel('Frequency',fontsize=tf)
    ax2.tick_params(axis='both', which='major', labelsize=tf-2)
    # Change legend order:
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,1]
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=tf-2)
    aligned_title(ax2, title=r'$\bf{B.}$ Random-game distribution',font=tf+4)

    fig.tight_layout()
    fig.savefig('plots/theory.png', dpi=res)


def _policy_theory_curve(ax, tf, branch_prob, max_game_length):
    pB = branch_prob
    iterations = 5 #8
    step_size = 4
    for i in range(iterations):
        state_probs = [1.0] 
        all_states = [] 
        n_turns = max_game_length - i*step_size
        for turn in range(n_turns): 
            branch_A = [pB *p for p in state_probs]
            branch_B = [(1.0-pB)*p for p in state_probs]
            state_probs = branch_A + branch_B
            all_states += state_probs
        all_states = [p for p in all_states if p >= 10**(-10)]

        all_states.sort(reverse = True)

        x = np.arange(len(all_states)) + 1
        y = all_states
        color = (n_turns - max_game_length + iterations*step_size) / (iterations*step_size)
        ax.scatter(x, y, s=3.0, color=cm.cividis(color))
        if i==0:
            # Fit:
            m =-1.
            c = np.log10(max(y)) + 0.3
            y_fit = 10 ** c * np.array(x) ** m
            ax.plot(x, y_fit, "black", markersize=1.0, label=r'$\alpha = 1$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('State rank',fontsize=tf)
    ax.set_ylabel('Frequency',fontsize=tf)
    ax.tick_params(axis='both', which='major', labelsize=tf-2)
    ax.legend(fontsize=tf)


def plot_appendix_theory_zipf(res: int = 300):  
    """ Plot frequency curves for an exponentially branching tree, with different policies."""
    print('~~~~~~~~~~~~~~~~~~~ Plotting Zipf law theory (appendix) ~~~~~~~~~~~~~~~~~~~')

    tf =14
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    max_game_length = 24
    norm = matplotlib.colors.Normalize(vmin=max_game_length- 4*5, vmax=max_game_length)
    sm = cm.ScalarMappable(cmap=plt.get_cmap('cividis'), norm=norm)
    cbar = fig.colorbar(sm, ax=axes[1])
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Game length', rotation=90, fontsize=tf)
    print('Calculating first plot...')
    _policy_theory_curve(axes[0],tf,0.7, max_game_length)
    aligned_title(axes[0], title=r'$\bf{A.}$ Lightly-skewed policy, $\boldsymbol{p}=(0.7,0.3)$',font=tf+2)
    print('Calculating second plot...')
    _policy_theory_curve(axes[1],tf,0.95, max_game_length)
    aligned_title(axes[1], title=r'$\bf{B.}$ Heavily-skewed policy, $\boldsymbol{p}=(0.95,0.05)$',font=tf+2)

    fig.tight_layout()
    print('Saving (can take a while)...')
    fig.savefig('plots/appendix_zipf_theory.png', dpi=res)
