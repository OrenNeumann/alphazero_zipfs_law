import numpy as np
from data_analysis.state_value.value_prediction import get_model_value_estimator
from tqdm import tqdm

def value_loss(env, path_model, board_count, info, num_chunks=20):
    """
    Calculate the value loss of a model on all states, sorted by rank.
    """
    serials = info['serials']
    real_values = info['values']
    model_values = get_model_value_estimator(env, path_model)
    sorted_serials = []
    z = []
    for key, _ in board_count.most_common():
        sorted_serials.append(serials[key])
        z.append(real_values[key])
    z = np.array(z)

    # Chunk data to smaller pieces to save memory:
    chunk_size = len(sorted_serials) // num_chunks
    data_chunks = [sorted_serials[i:i + chunk_size] for i in range(0, len(sorted_serials), chunk_size)]
    vl = []
    for chunk in tqdm(data_chunks, desc='Estimating model loss'):
        vl.append(model_values(chunk))
    v = np.concatenate(vl)

    return (z - v) ** 2