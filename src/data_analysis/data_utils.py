import collections
import numpy as np
from tqdm import tqdm


def sort_by_frequency(data: dict, counter: collections.Counter) -> np.ndarray:
    """ Sort any state-data by descending order of frequency in counter."""
    sorted_data = np.zeros(len(counter))
    for idx, entry in enumerate(tqdm(counter.most_common())):
        key = entry[0]
        sorted_data[idx] = data[key]
    return sorted_data


