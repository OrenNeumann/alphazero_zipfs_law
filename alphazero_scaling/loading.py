import pyspiel
import json
import collections
from open_spiel.python.algorithms.alpha_zero import alpha_zero as az_lib

"""
Code used for the AlphaZero scaling laws paper. 
Mostly functions that load saved models.
"""

class Config(collections.namedtuple(
    "Config", [
        "game",
        "MC_matches",
        "path",
        "path_model_1",
        "path_model_2",
        "checkpoint_number_1",
        "checkpoint_number_2",
        "use_solver",
        "use_two_solvers",
        "solver_1_temp",
        "solver_2_temp",

        "logfile",
        "learning_rate",
        "weight_decay",
        "temperature",
        "evaluators",
        "evaluation_games",
        "evaluation_window",

        "uct_c",
        "max_simulations",
        "policy_alpha",

        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",

        "quiet",
    ])):

    pass

def load_config(dir_name, version='training'):
    """ Create a config object from a json file.
        'version' controls the type of Config object:
        'training': from the alpha_zero library;
        'matches': from this code.
    """
    # path = './' + dir_name + '/config.json'  # Old code (office computer)
    path = dir_name + '/config.json'  # New code (cluster)
    with open(path) as f:
        data = json.load(f)
    if version == 'training':
        config = az_lib.Config(**data)
    elif version == 'matches':
        config = Config(**data)
    else:
        raise Exception('Not a valid Config type.')
    return config


def get_model_params(path):
    conf = load_config(path, version='training')
    params = {
        'nn_model': conf.nn_model,
        'nn_width': conf.nn_width,
        'nn_depth': conf.nn_depth
    }
    return params


def load_model_from_checkpoint(config, checkpoint_number=None, path=None):
    """
    :param config: A Config object.
    :param checkpoint_number: Number of checkpoint to load. If none given, will
                load the last checkpoint saved.
    :param path: Path to the saved model. If nothing is given, will
                load from config.path .
    :return: A NN model with weights loaded from the latest checkpoint.
    """
    if path is None:
        path = config.path

    # Make some changes to the config object
    name = config.game
    game = pyspiel.load_game(name)
    params = get_model_params(path)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),  # Input size
        output_size=game.num_distinct_actions(),  # Output size
        **params)  # Model parameters (type, width, depth)

    if checkpoint_number is None:
        # Get the latest checkpoint in the log and load it to a model
        variables = {}
        with open(path + 'checkpoint') as f:
            for line in f:
                name, value = line.split(": ")
                variables[name] = value
        checkpoint_name = variables['model_checkpoint_path'][1:-2]
    else:
        # Get the specified model number (on the cluster:
        checkpoint_name = path + 'checkpoint-' + str(checkpoint_number)

    checkpoint_path = checkpoint_name
    model = az_lib._init_model_from_config(config)
    model.load_checkpoint(checkpoint_path)

    return model
