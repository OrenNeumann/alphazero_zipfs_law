from src.alphazero_scaling.elo.plotting import plot_oware_size_scaling, plot_new_size_scaling
from src.plotting.plot_scripts.value_loss import connect4_loss_plots, oware_value_loss
from src.plotting.plot_scripts.game_turns import plot_game_turns
from src.plotting.plot_scripts.elo_curves import plot_scaling_failure
from src.plotting.plot_scripts.appendix import board_positions as board
from src.plotting.plot_scripts.zipf_law_theory import plot_zipf_law_theory, plot_appendix_theory_zipf
from src.plotting.plot_scripts.zipf_curves import plot_main_zipf_curves, plot_appendix_zipf_curves
from src.plotting.plot_scripts.temperature import plot_temperature_curves
from src.plotting.plot_scripts.appendix.policy_degradation import plot_policy_degradation
from src.plotting.plot_scripts.appendix.capture_difference import plot_capture_differences

############################################
# write how much time on what kind of hardware is needed for all experiments.
# CPU cores, RAM
############################################

"""
Re-create all plots from the paper.
If 'load_data' is False, generates all plot data from scratch. This will take a while.

Note regarding generating data from scratch:
Some experiments require importing or installing 3rd party data and software manually:
- Almost all experiments require agent training/inference data, and the trained agents. 
    These can be generated with OpenSpiel by the code available here:
    https://github.com/OrenNeumann/AlphaZero-scaling-laws
- Experiments using the Connect Four solver ('connect4_loss_plots', 'plot_policy_degradation') require installing the solver available here:
    https://github.com/PascalPons/connect4
    The opening book, available as a release in the github repo, should be saved in the running directory.
- The alpha-beta pruning experiment in 'connect4_loss_plots' requires that the solver opening book is NOT present in the 
    parent dir (forcing the solver to perform a full search without prior data).
- 'plot_appendix_zipf_curves' should run before 'plot_main_zipf_curves'.

"""
load_data = True
res = 50

#plot_appendix_zipf_curves(load_data, res=50) # Generates data for plot_main_zipf_curves

# Main paper:
#plot_main_zipf_curves(res=300)
#plot_zipf_law_theory(load_data, res=50)
plot_temperature_curves(load_data, res=100)
#connect4_loss_plots(load_data, res=50)
#plot_scaling_failure(load_data, res=50)
#plot_game_turns(res=50)
#oware_value_loss(load_data, res=300)

# Appendix:
#plot_appendix_theory_zipf(res=300)
#plot_policy_degradation(load_data, res=50)
#board.plot_oware(res=50)
#board.plot_checkers(res=50)
#plot_capture_differences(load_data, res)