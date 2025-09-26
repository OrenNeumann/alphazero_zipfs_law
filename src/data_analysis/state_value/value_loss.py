import numpy as np
from src.data_analysis.state_value.value_prediction import get_model_value_estimator
from src.data_analysis.state_frequency.state_counter import StateCounter
from tqdm import tqdm
from src.data_analysis.state_value.value_prediction import get_solver_value_estimator
from multiprocessing import Pool, cpu_count


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

def solver_loss(env, path_model: str,
               state_counter: StateCounter,
               checkpoint_number: int = 10_000,
               num_chunks: int = 40,
               chunk_size_solver: int = 100) -> np.ndarray:
    """
    Calculate the ground truth loss of a model on all states, sorted by rank,
    using the solver values as ground truth labels.
    Parallelizes solver_value calls in batches.
    """
    model_values = get_model_value_estimator(env, path_model, checkpoint_number=checkpoint_number)
    solver_value = get_solver_value_estimator(env)

    # All serials sorted by frequency rank
    sorted_serials = [state_counter.serials[key] for key, _ in state_counter.frequencies.most_common()]

    # --- Parallel solver evaluation ---
    def process_chunk(chunk):
        return [solver_value(s) for s in chunk]

    # Split into batches of `chunk_size_solver`
    solver_chunks = [sorted_serials[i:i + chunk_size_solver] 
                     for i in range(0, len(sorted_serials), chunk_size_solver)]

    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(tqdm(pool.imap(process_chunk, solver_chunks), 
                            total=len(solver_chunks), 
                            desc="Calculating solver values"))
    z = np.array([val for batch in results for val in batch])

    # --- Model evaluation (already chunked for memory) ---
    chunk_size = len(sorted_serials) // num_chunks
    data_chunks = [sorted_serials[i:i + chunk_size] 
                   for i in range(0, len(sorted_serials), chunk_size)]

    v = np.concatenate([model_values(chunk) 
                        for chunk in tqdm(data_chunks, desc="Estimating model loss")])

    return (z - v) ** 2
