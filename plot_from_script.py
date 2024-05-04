from src.alphazero_scaling.elo.plotting import plot_oware_size_scaling, plot_new_size_scaling
from src.plotting.plot_scripts.value_loss import connect4_loss_plots, oware_value_loss
from src.plotting.plot_scripts.game_turns import plot_game_turns
from src.plotting.plot_scripts.elo_curves import plot_scaling_failure
from src.plotting.plot_scripts import board_positions as board
from src.plotting.plot_scripts.zipf_law_theory import plot_zipf_law_theory
from src.plotting.plot_scripts.zipf_curves import plot_zipf_curves, plot_temperature_curves
from src.plotting.plot_scripts.policy_degradation import plot_policy_degradation

"""
Re-create all plots from the paper.
If load_data is False, generates all the plot data from scratch. This will take a while.
"""
load_data = True
res = 50

# Main paper:
#plot_zipf_law_theory(load_data=True, res=50)
#plot_temperature_curves(load_data=True, res=50)
#connect4_loss_plots(load_data=True, res=50)
#plot_scaling_failure(load_data=True, res=50)
#plot_game_turns(res=50)
#oware_value_loss(res=50)

# Appendix:
#plot_zipf_curves(load_data=True, res=50)
plot_policy_degradation(load_data=True, res=300)
#board.plot_oware(res=50)
#board.plot_checkers(res=50)