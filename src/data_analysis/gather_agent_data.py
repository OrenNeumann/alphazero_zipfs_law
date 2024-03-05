from collections import Counter
from src.data_analysis.game_data_analysis import process_games
from src.general.general_utils import models_path, game_path


def gather_data(env: str, labels: list[int], max_file_num: int = 1, save_serial: bool = False,
                save_turn_num: bool = False, save_value: bool = False) -> tuple[Counter, dict]:
    """ Gather data from multiple agents, aggregated together.
        Counts and averages are taken over all agents in 'labels'.
        Currently just uses agent copy num. 0."""
    print('Collecting ' + env + ' games:')
    path = models_path() + game_path(env) + 'q_'
    board_counter = Counter()
    serial_states = dict()
    turns_played = dict()
    turns_to_end = dict()
    values = dict()
    for label in labels:
        num = str(label)
        print("label " + num)
        agent_path = path + num + '_2'
        temp_counter, info = process_games(env, agent_path, max_file_num=max_file_num, save_serial=save_serial,
                                           save_turn_num=save_turn_num, save_value=save_value)
        # add counts to the counter:
        board_counter.update(temp_counter)

        if save_serial:
            serial_states.update(info['serials'])
        if save_turn_num:
            _update_dict(turns_played, info['turns_played'])
            _update_dict(turns_to_end, info['turns_to_end'])
        if save_value:
            _update_dict(values, info['values'])

    if save_turn_num:
        for key, n in board_counter.items():
            turns_played[key] /= n
            turns_to_end[key] /= n
    if save_value:
        for key, n in board_counter.items():
            values[key] /= n
    total_info = {'serials': serial_states,
                  'turns_played': turns_played,
                  'turns_to_end': turns_to_end,
                  'values': values}

    return board_counter, total_info


def _update_dict(dictionary: dict, data: dict):
    """ Updates 'dictionary' by adding the values in 'data' to the existing values.
        Similar to updating a Counter, but for non-integer numbers."""
    temp_data = dict()
    for key in data.keys():
        temp_data[key] = data[key] + dictionary.get(key, 0)
    dictionary.update(temp_data)
