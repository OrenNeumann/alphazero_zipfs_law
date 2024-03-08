from collections import Counter
from src.data_analysis.game_data_analysis import process_games, noramlize_info
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.general.general_utils import models_path, game_path


def gather_data(env: str, labels: list[int], max_file_num: int = 1, save_serial: bool = False,
                save_turn_num: bool = False, save_value: bool = False) -> tuple[Counter, dict]:
    """ Gather data from multiple agents, aggregated together.
        Counts and averages are taken over all agents in 'labels'.
        Currently just uses agent copy num. 0."""
    print('Collecting ' + env + ' games:')
    path = models_path() + game_path(env) + 'q_'
    state_counter = StateCounter(env, save_serial=save_serial, save_turn_num=save_turn_num, save_value=save_value)
    for label in labels:
        num = str(label)
        print("label " + num)
        agent_path = path + num + '_2/'
        state_counter.collect_data(path=agent_path, max_file_num=max_file_num)

    state_counter.normalize_counters()

    return state_counter


def _update_dict(dictionary: dict, data: dict):
    """ Updates 'dictionary' by adding the values in 'data' to the existing values.
        Similar to updating a Counter, but for non-integer numbers."""
    temp_data = dict()
    for key in data.keys():
        temp_data[key] = data[key] + dictionary.get(key, 0)
    dictionary.update(temp_data)
