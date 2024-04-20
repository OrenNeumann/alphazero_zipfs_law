import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from src.plotting.plot_utils import aligned_title
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.general.general_utils import models_path

plt.style.use(['science','nature','grid'])

def _generate_zipf_curves(env, models):
    print('Generating Zipf curves for ' + env)
    state_counter = StateCounter(env=env)
    freqs = {model: [] for model in models}
    for model in models:
        path = models_path() + env + '_10000/' + model + '/'
        state_counter.collect_data(path=path, max_file_num=39)
        freqs[model] = np.array([item[1] for item in state_counter.frequencies.most_common()])
    with open(f'../plot_data/zipf_curves/zipf_curves_{env}.pkl', 'wb') as f:
        pickle.dump(freqs, f)

def plot_zipf_curves(load_data=True):
    """
    Plot Zipf curves.
    """
    models = ['q_0_0', 'q_2_0', 'q_4_0', 'q_6_0']
    envs = ['connect_four', 'pentago', 'oware', 'checkers']
    if not load_data:
        for env in envs:
            _generate_zipf_curves(env, models)

    tf =12
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 3))

    print('Plotting Zipf curves')
    with open('../plot_data/zipf_curves/zipf_curves.pkl', 'rb') as f:
        zipf_curves = pickle.load(f)
    for i, ax in enumerate(axes):
        x, y = zipf_curves[label]
        ax.plot(x, y, label=str(label))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=tf-2)
        ax.set_xlabel('State rank',fontsize=tf)
        ax.set_ylabel('Frequency',fontsize=tf)
        ax.legend(fontsize=tf-2)

    aligned_title(ax, 'Zipf curves', tf)

    fig.tight_layout()
    fig.savefig('plots/zipf_curves.png', dpi=300)