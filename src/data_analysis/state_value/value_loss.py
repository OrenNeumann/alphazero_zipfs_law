import numpy as np
from src.data_analysis.state_value.value_prediction import get_model_value_estimator
from src.data_analysis.state_frequency.state_counter import StateCounter
from tqdm import tqdm
from src.data_analysis.state_value.value_prediction import get_solver_value_estimator


def value_loss(env, path_model: str,
               state_counter: StateCounter,
               checkpoint_number: int = 10_000,
               num_chunks: int = 40,
               values=None) -> np.ndarray:
    """
    Calculate the value loss of a model on all states, sorted by rank.
    Uses 'values' as labels if given, otherwise uses the values saved in state_counter.
    """
    if values is None:
        values = state_counter.values
    if len(state_counter.serials) == 0 or len(values) == 0:
        raise ValueError('Serials and/or values counter are empty.\n'
                         'Did you forget to call StateCounter with save_serial=True, save_value=True?')
    model_values = get_model_value_estimator(env, path_model, checkpoint_number=checkpoint_number)
    sorted_serials = []
    z = []
    for key, _ in state_counter.frequencies.most_common():
        sorted_serials.append(state_counter.serials[key])
        z.append(values[key])
    z = np.array(z)

    # Chunk data to smaller pieces to save memory:
    chunk_size = len(sorted_serials) // num_chunks
    data_chunks = [sorted_serials[i:i + chunk_size] for i in range(0, len(sorted_serials), chunk_size)]
    vl = []
    for chunk in tqdm(data_chunks, desc='Estimating model loss'):
        vl.append(model_values(chunk))
    v = np.concatenate(vl)

    return (z - v) ** 2

def _calculate_solver_values_chunk(args):
    """Helper function to calculate solver values for a chunk of serials."""
    env, serials_chunk = args
    solver_value = get_solver_value_estimator(env)
    return [solver_value(serial) for serial in serials_chunk]

def solver_loss(env, path_model: str,
               state_counter: StateCounter,
               checkpoint_number: int = 10_000,
               num_chunks: int = 40) -> np.ndarray:
    """
    Calculate the ground truth loss of a model on all states, sorted by rank,
    using the solver values as ground truth labels.
    """
    model_values = get_model_value_estimator(env, path_model, checkpoint_number=checkpoint_number)
    sorted_serials = []
    z = []
    for key, _ in tqdm(state_counter.frequencies.most_common(), desc='calculating solver values'):
        serial = state_counter.serials[key]
        sorted_serials.append(state_counter.serials[key])
        solver_value = get_solver_value_estimator(env)
        z.append(solver_value(serial))
    z = np.array(z)

    # Chunk data to smaller pieces to save memory:
    chunk_size = len(sorted_serials) // num_chunks
    data_chunks = [sorted_serials[i:i + chunk_size] for i in range(0, len(sorted_serials), chunk_size)]
    vl = []
    for chunk in tqdm(data_chunks, desc='Estimating model loss'):
        vl.append(model_values(chunk))
    v = np.concatenate(vl)

    return (z - v) ** 2