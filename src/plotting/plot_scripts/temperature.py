import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scienceplots
import pickle
from tqdm import tqdm
from itertools import combinations
from src.plotting.plot_utils import aligned_title
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.alphazero_scaling.elo.utils import PlayerNums, BayesElo


plt.style.use(['science','nature','grid'])

TEMPERATURES = np.array([0.07, 0.1, 0.14, 0.2, 0.25, 0.32, 0.45, 0.6, 0.8, 1, 1.4, 2, 3, 5, 0.04, 0.02, 0.01])

def _generate_temperature_zipf_curves(k: int): 
    max_q = 6
    n_copies = 3
    nets = []
    for i in range(max_q + 1):
        for j in range(n_copies):
            nets.append('q_' + str(i) + '_' + str(j))
    n = len(nets)
    counter = StateCounter(env='connect_four')
    freqs = dict()
    path_dir = f'../plot_data/temperature/game_data/temp_num_{k}'
    for pair in tqdm(list(combinations(range(n), 2)), desc=f'Collecting T={TEMPERATURES[k]} matches'): 
        path = path_dir + '/' + nets[pair[0]] + '_vs_' + nets[pair[1]] + '/'
        counter.collect_data(path=path, max_file_num=80, quiet=True, matches=True)
    freqs= np.array([item[1] for item in counter.frequencies.most_common()])
    with open(f'../plot_data/temperature/zipf_curves/temp_num_{k}.pkl', 'wb') as f:
        pickle.dump(freqs, f)


def plot_temperature_curves(load_data=True, res=300):
    print('~~~~~~~~~~~~~~~~~~~ Plotting temperature curves ~~~~~~~~~~~~~~~~~~~')
    sorted_t = np.argsort(TEMPERATURES)
    log_t = np.log(TEMPERATURES)
    color_nums = (log_t - log_t.min()) / (log_t.max() - log_t.min()) 
    tf =12
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1.2, 1, 1]})
    par = np.load('src/config/parameter_counts/connect_four.npy')
    zipf_exponents = np.zeros(len(TEMPERATURES))

    print('Plotting Connect Four Zipf curves at different temperatures.')
    if not load_data:
        for ind in sorted_t:
            _generate_temperature_zipf_curves(ind)
    for ind in tqdm(sorted_t[::-1], desc='Plotting Zipf curves'):
        with open(f'../plot_data/temperature/zipf_curves/temp_num_{ind}.pkl', 'rb') as f:
            zipf_curve = pickle.load(f)
        ###
        #zipf_curve = zipf_curve[:np.argmax(zipf_curve == 1)] #prune
        ###
        x = np.arange(len(zipf_curve))+1
        axs[0].scatter(x,zipf_curve, color=cm.plasma(color_nums[ind]), s=40 / (10 + x))

        #fitting between counts 200 and 4:
        low = np.argmax(zipf_curve < 200)
        up = np.argmax(zipf_curve == 4)
        x_nums = np.arange(up)[low:]
        [m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(zipf_curve[low:up]), deg=1, w=2 / x_nums)
        zipf_exponents[ind]=-m

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('State rank',fontsize=tf)
    axs[0].set_ylabel('Frequency',fontsize=tf)
    axs[0].tick_params(axis='both', which='major', labelsize=tf-2)
    aligned_title(axs[0], r"$\bf{A.}$ Inference Zipf's law", tf+4)

    ##############################################################
    print('Plotting size scaling at different temperatures')
    elo_exponents = np.zeros(len(TEMPERATURES))
    n=0
    for ind in sorted_t:
        n+=1
        print(f'({n}/{len(TEMPERATURES)}) Temperature: {TEMPERATURES[ind]}')
        r = BayesElo()
        agents = PlayerNums()
        matches = np.load(f'../matches/temperature_scaling/connect_four_temp_num_{ind}/matrix.npy')
        sizes = 7
        copies = 4
        for i, j in combinations(range(len(matches)), 2):
            r.add_match(i, j, p=matches[i, j])
        for i in range(sizes):
            for j in range(copies):
                agents.add(f'q_{i}_{j}', 10000)
        elo = r.extract_elo(agents)

        elo_scores = []
        elo_stds = []
        all_scores = []
        all_params = []
        for i in range(sizes):
            elo_scores.append(np.mean([elo[f'q_{i}_{j},10000'] for j in range(copies)]) )
            elo_stds.append(np.std([elo[f'q_{i}_{j},10000'] for j in range(copies)]) )
            for j in range(copies):
                all_scores.append(elo[f'q_{i}_{j},10000'])
                all_params.append(par[i])
        elo_scores = np.array(elo_scores)
        # Set Elo score range
        elo_scores += 100 - elo_scores.min()
        axs[1].errorbar(par[:-2], elo_scores[:-2], yerr=[elo_stds[:-2], elo_stds[:-2]], fmt='-o', 
                    color=cm.plasma(color_nums[ind]), linewidth=0.5, markersize=0.5)
        #fitting:
        [m, c] = np.polyfit(np.log10(all_params[:-2*copies]), all_scores[:-2*copies], 1)
        elo_exponents[ind] = m/400
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Neural-net parameters',fontsize=tf)
    axs[1].set_ylabel('Elo',fontsize=tf)
    axs[1].tick_params(axis='both', which='major', labelsize=tf-2)
    aligned_title(axs[1], r"$\bf{B.}$ Size scaling law", tf+4)
    
    ##############################################################
    print('Plotting exponents relation')

    # plotting low-T data:
    indices = sorted_t[:9]
    axs[2].plot(zipf_exponents[indices], elo_exponents[indices],  
                    markersize=0, linestyle='--', color='gray')
    axs[2].scatter(zipf_exponents[indices], elo_exponents[indices], c=cm.plasma(color_nums[indices]), s=60)
    axs[2].set_xlabel('Zipf exponent (tail)',fontsize=tf)
    axs[2].set_ylabel('Elo exponent',fontsize=tf)
    axs[2].tick_params(axis='both', which='major', labelsize=tf-2)
    aligned_title(axs[2], r"$\bf{C.}$ Exponent correlation", tf+4)

    # Colorbar:
    norm = matplotlib.colors.LogNorm(vmin=TEMPERATURES.min(), vmax=TEMPERATURES.max())
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=norm)
    cbar = fig.colorbar(sm, ax=axs[2])
    cbar.ax.tick_params(labelsize=tf)
    cbar.ax.set_ylabel('Temperature', rotation=90, fontsize=tf)

    fig.tight_layout()
    print('Saving figure (can take a while)...')
    fig.savefig('./plots/temperature_curves.png', dpi=res)

    ##############################################################
    print('Plotting exponents relation (appendix)')
    fig = plt.figure(figsize=(12, 6))

    # plotting all-T data:
    plt.scatter(zipf_exponents, elo_exponents, c=cm.plasma(color_nums), s=70)
    plt.axvline(x=1, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Zipf exponent (tail)',fontsize=tf+8)
    plt.ylabel('Elo exponent',fontsize=tf+8)
    plt.tick_params(axis='both', which='major', labelsize=tf+6)
    plt.annotate('Fig. 3B cutoff', xy=(1, 0.6), xytext=(2.2, 0.5), arrowprops=dict(arrowstyle='->'), fontsize=tf+8)
    print(zipf_exponents)
    print(elo_exponents)
    
    # Colorbar:
    cbar = fig.colorbar(sm)
    cbar.ax.tick_params(labelsize=tf+6)
    cbar.ax.set_ylabel('Temperature', rotation=90, fontsize=tf+8)

    fig.tight_layout()
    fig.savefig('./plots/exponent_correlation.png', dpi=res)
