import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from src.plotting.plot_utils import aligned_title


plt.style.use(['science','nature','grid'])

def _generate_zipf_curves():


def plot_zipf_curves(load_data=True):
    """
    Plot Zipf curves.
    """
    
    if not load_data:
        _generate_zipf_curves()

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