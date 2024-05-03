import numpy as np
import pickle
from src.data_analysis.state_value.solver_values import solver_optimal_moves
from src.general.general_utils import models_path, game_path
from src.data_analysis.state_value.value_loss import value_loss
from src.plotting.plot_utils import BarFigure, incremental_bin, smooth, gaussian_average
from src.data_analysis.state_frequency.state_counter import StateCounter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Plot the effect of temperature on agent's strategy quality.
Calculates the probability that an agent takes an optimal move.
(run plot_solver_value_loss.py first)
"""

def save_optimal_moves():
    with open('../plot_data/solver/state_counter.pkl', 'rb') as f:
        state_counter = pickle.load(f)
    chunk_size = 100
    optimal_moves = dict()
    for i in tqdm(range(0, len(state_counter.frequencies), chunk_size), desc="Estimating optimal moves"): 
        keys = list(state_counter.frequencies.keys())[i:i+chunk_size]
        vals = solver_optimal_moves([state_counter.serials[key] for key in keys])
        optimal_moves.update({key: val for key, val in zip(keys, vals)})
    with open('../plot_data/solver/optimal_moves.pkl', 'wb') as f:
        pickle.dump(optimal_moves, f)


