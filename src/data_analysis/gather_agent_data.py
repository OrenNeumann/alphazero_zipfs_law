from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_frequency.counting_mods.cutoff_counter import CutoffCounter
from src.general.general_utils import models_path, game_path


def gather_data(env: str,
                labels: list[int],
                cutoff=None,
                max_file_num: int = 1,
                save_serial: bool = False,
                save_turn_num: bool = False,
                save_value: bool = False) -> StateCounter:
    """ Gather data from multiple agents, aggregated together.
        Counts and averages are taken over all agents in 'labels'.
        Currently just uses agent copy num. 0."""
    print('Collecting ' + env + ' games:')
    path = models_path() + game_path(env) + 'q_'
    if cutoff is None:
        state_counter = StateCounter(env=env, save_serial=save_serial, save_turn_num=save_turn_num,
                                     save_value=save_value, cut_early_games=False)
    else:
        state_counter = CutoffCounter(cutoff=cutoff, env=env, save_serial=save_serial, save_turn_num=save_turn_num,
                                      save_value=save_value, cut_early_games=False) ####
    for label in labels:
        num = str(label)
        print("label " + num)
        agent_path = path + num + '_0'
        state_counter.collect_data(path=agent_path, max_file_num=max_file_num)

    state_counter.normalize_counters()
    return state_counter


