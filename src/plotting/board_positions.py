from cairosvg import svg2png
import chess.svg
from src.data_analysis.gather_agent_data import gather_data
import pickle
import matplotlib.pyplot as plt
import os
import scienceplots

plt.style.use(['science','nature'])

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


def create_subtitle(fig, grid, title):
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontsize=18)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

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
    early_turns = [0,9,99,999]#[0,10,100,1000]
    late_turns = [39, 140, 556, 3345]
    for index in early_turns + late_turns:
        plot_checkers_state(keys[index], str(index))

    print('Plotting subplots')

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Figure title')
    names = early_turns + late_turns
    row_titles = ['Checkers Board states', 'Late-game board states']
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=2, ncols=1)
    png_dir = 'plots/board_positions/'
    # Iterate over each name
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(row_titles[row], fontsize=18)
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=4)
        for col, ax in enumerate(axs):
            name = names[row*4+col]
            png_file = os.path.join(png_dir, f'checkers-board_{name}.png')
            img = plt.imread(png_file)
            ax.imshow(img)
            ax.axis('off') 
            ax.set_title(f'State #{name+1}', fontsize=16)
    plt.tight_layout()
    fig.savefig('./plots/checkers_positions.png', dpi=600)
    """
    # Create a Matplotlib figure and axes
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    png_dir='plots/board_positions/'
    # Iterate over each name
    for i, name in enumerate(early_turns + late_turns):
        # Calculate the subplot position
        row = i // 4
        col = i % 4
        # Load the PNG file
        png_file = os.path.join(png_dir, f'checkers-board_{name}.png')
        img = plt.imread(png_file)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')  # Hide axis
        axs[row, col].set_title(f'State {name}', fontsize=16)
    grid = plt.GridSpec(2, 4)
    create_subtitle(fig, grid[0, ::], 'Checkers board states')
    create_subtitle(fig, grid[1, ::], 'Late-game board states')
    plt.tight_layout()
    fig.savefig('./plots/checkers_positions.png', dpi=600)"""