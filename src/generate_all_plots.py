from src.plotting.plot_scripts.value_loss import connect4_loss_plots, oware_value_loss
from src.plotting.plot_scripts.game_turns import plot_game_turns
from src.plotting.plot_scripts.elo_curves import plot_scaling_failure
from src.plotting.plot_scripts.appendix import board_positions as board
from src.plotting.plot_scripts.zipf_law_theory import plot_zipf_law_theory, plot_appendix_theory_zipf
from src.plotting.plot_scripts.zipf_curves import plot_main_zipf_curves, plot_appendix_zipf_curves
from src.plotting.plot_scripts.temperature import plot_temperature_curves
from src.plotting.plot_scripts.appendix.policy_degradation import plot_policy_degradation
from src.plotting.plot_scripts.appendix.capture_difference import plot_capture_differences


"""
Re-create all plots from the paper.
If 'load_data' is False, generates all plot data from scratch (takes a while).
"""
load_data = True

# 'plot_appendix_zipf_curves' should run before 'plot_main_zipf_curves'
plot_appendix_zipf_curves(load_data)

# Main paper:
plot_main_zipf_curves()
plot_zipf_law_theory(load_data)
plot_temperature_curves(load_data)
connect4_loss_plots(load_data)
plot_scaling_failure(load_data)
plot_game_turns()
oware_value_loss(load_data)

# Appendix:
plot_appendix_theory_zipf()
plot_policy_degradation(load_data)
board.plot_oware()
board.plot_checkers()
plot_capture_differences(load_data)