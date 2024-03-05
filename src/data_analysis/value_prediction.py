from src.alphazero_scaling.solver_bot import connect_four_solver
from src.alphazero_scaling.loading import load_model_from_checkpoint, load_config
import numpy as np
import numpy.typing as npt
import pyspiel


def _get_game(env):
    """Return a pyspiel game"""
    if env == 'connect4':
        return pyspiel.load_game('connect_four')
    elif env == 'pentago':
        return pyspiel.load_game('pentago')
    elif env == 'oware':
        return pyspiel.load_game('oware')
    elif env == 'checkers':
        return pyspiel.load_game('checkers')
    else:
        raise NameError('Environment ' + str(env) + ' not supported.')


def get_model_value_estimator(env: str, config_path: str):
    """ Creates an estimator function that uses a trained model to calculate state values.

        :param env: Game environment.
        :param config_path: Path to trained model.
    """
    game = _get_game(env)
    config = load_config(config_path)
    model = load_model_from_checkpoint(config=config, path=config_path, checkpoint_number=10_000)

    def model_value(serial_states: list[str]) -> npt.NDArray[np.float64]:
        """
        Calculate value estimations of the model on a list of board states.
        Note: this will crash if serial_states is too large and the network is
        too big. If you get a segmentation fault, call this function on smaller
        chunks of data.
        """

        observations = []
        masks = []
        for i, serial in enumerate(serial_states):
            state = game.deserialize_state(serial)
            observations.append(state.observation_tensor())
            masks.append(state.legal_actions_mask())
        values = model.inference(observations, masks)[0]

        return values.flatten()

    return model_value


def get_solver_value_estimator(env: str):
    """
    Create an estimator based on the Connect 4 solver.
    Only supported for Connect 4, could add Pentago.

    :param env: Game environment.
    :return: value_loss function.
    """

    if env != 'connect4':
        raise NameError('Game name provided not supported: ' + env)

    # Load solver from solver_bot
    game = pyspiel.load_game('connect_four')

    def solver_value(serial_state: str) -> float:
        """
        Calculate ground truth value given by the solver.
        """
        state = game.deserialize_state(serial_state)
        if state.is_terminal():
            # if the state is terminal, the model has nothing to predict.
            raise Exception('Terminal state encountered')
        solver_v, p = connect_four_solver(state, full_info=True)
        return solver_v

    return solver_value
