import numpy as np
import subprocess
from subprocess import PIPE
import pyspiel
from tqdm import tqdm
import multiprocessing as mp
from src.data_analysis.state_frequency.state_counter import StateCounter
from src.data_analysis.state_value.value_prediction import get_model_value_estimator


GAME = pyspiel.load_game('connect_four')

def connect_four_solver(states: list, batch_size: int = None, show_progress: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ Produces an optimal-scores vector for a game state.
        Takes a list of states and returns for each a mask of legal moves, plus a vector
        of scores for each move. The scores are integers indicating how early the game will end, with 
        a +/- sign for win/loss. Higher score = better move.

        This code uses the open source solver available in:
        https://connect4.gamesolver.org/

        In order to use this, download the Github repo:
        https://github.com/PascalPons/connect4
        Hit 'make' and copy the call_solver.sh script into the folder.
        Then download the openings book here: (7x6.book)
        https://github.com/PascalPons/connect4/releases/tag/book
        and place it in the working directory.
        """
    if not states:
        return [], []
    
    # Set default batch size based on CPU count
    if batch_size is None:
        batch_size = max(4, min(50, mp.cpu_count() * 2))
    
    all_masks = []
    all_scores = []
    
    # Process in batches
    batch_iterator = range(0, len(states), batch_size)
    if show_progress and len(states) > batch_size:
        batch_iterator = tqdm(batch_iterator, desc=f"Processing {len(states)} states")
    
    for i in batch_iterator:
        batch_states = states[i:i + batch_size]
        
        # Original logic for this batch
        procs = []
        masks = []
        moves_list = []  # Store moves for later use
        
        for state in batch_states:
            # The solver starts counting moves from 1:
            moves = ''.join([str(action.action + 1) for action in state.full_history()])
            moves_list.append(moves)
            masks.append(state.legal_actions_mask())
            procs.append(subprocess.Popen(["src/alphazero_scaling/connect4-master/call_solver.sh", moves], stdout=PIPE, stderr=PIPE))
        
        scores = []
        for j, proc in enumerate(procs):
            out, _ = proc.communicate()
            out = out.split()
            moves = moves_list[j]  # Use stored moves
            if moves == "":
                scores.append(np.array(out, dtype=int))
            else:  # ignore 1st output (=moves):
                scores.append(np.array(out[1:], dtype=int))
        
        all_masks.extend(masks)
        all_scores.extend(scores)
    
    return all_masks, all_scores

def solver_values(serial_states: list[str], batch_size: int = None, show_progress: bool = True) -> list[float]:
    """
    Calculate ground truth value given by the solver.
    Returns the value for the first player.
    """
    states = [GAME.deserialize_state(serial) for serial in serial_states]
    solver_v = _get_values(states, batch_size=batch_size, show_progress=show_progress)
    player_0_values = [solver_v[i] * (1 - 2 * state.current_player()) for i, state in enumerate(states)]
    return player_0_values

def _get_values(states: list, batch_size: int = None, show_progress: bool = True):
    masks, scores = connect_four_solver(states, batch_size=batch_size, show_progress=show_progress)
    values = []
    for i, score_vec in enumerate(scores):
        mask = masks[i]
        p = np.extract(mask, score_vec)
        # v=[1,0,-1] for win, draw, loss
        values.append(max(np.sign(p)))
    return values

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
    z = solver_values([key for key, _ in state_counter.frequencies.most_common()])
    z = np.array(z)
    
    # Chunk data to smaller pieces to save memory:
    chunk_size = len(sorted_serials) // num_chunks
    data_chunks = [sorted_serials[i:i + chunk_size] for i in range(0, len(sorted_serials), chunk_size)]
    vl = []
    for chunk in tqdm(data_chunks, desc='Estimating model loss'):
        vl.append(model_values(chunk))
    v = np.concatenate(vl)

    return (z - v) ** 2