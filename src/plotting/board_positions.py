from cairosvg import svg2png
import chess.svg
from src.data_analysis.gather_agent_data import gather_data
import pickle
from tqdm import tqdm

checkers_symbols = {'o': 'P', '+': 'p', '8': 'K', '*': 'k'}

def checkers_to_fen(checkers_position):
    """
    Convert a checkers position string to a chess FEN string.
    """
    board = checkers_position.split('\n')[:-2]
    fen_rows = []
    for row in range(8):
        fen_row = ''
        empty = 0
        for col in range(1,9,1):
            piece = board[row][col]
            if piece == '.':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += checkers_symbols[piece]
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    fen = '/'.join(fen_rows)
    return f"{fen} w - - 0 1"


def plot_checkers_state(state, name):
    fen = checkers_to_fen(str(state))
    board = chess.Board(fen)
    svg_text = chess.svg.board(
        board,
        size=350)
    svg2png(bytestring=svg_text, write_to='plots/board_positions/checkers-board_'+name+'.png')


def plot_checkers():
    save_data = False
    if save_data:
        state_counter = gather_data('checkers', [0,1,2,3,4,5,6], max_file_num=10, save_turn_num=True)
        state_counter.prune_low_frequencies(threshold=10)
        with open('../plot_data/board_positions/checkers_counter.pkl', 'wb') as f:
            pickle.dump(state_counter, f)
    print('Loading data')
    with open('../plot_data/board_positions/checkers_counter.pkl', 'rb') as f:
        state_counter = pickle.load(f)
    #turn_mask = state_counter.late_turn_mask(threshold=40)
    keys = [i for i,_ in state_counter.frequencies.most_common()]
    early_turns = [0,10,100,1000]
    late_turns = [39, 140, 556, 3345]

    for index in tqdm(early_turns + late_turns):
        plot_checkers_state(keys[index], str(index))
