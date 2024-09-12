import matplotlib
import matplotlib.figure
from cairosvg import svg2png
import chess.svg
from src.data_analysis.gather_agent_data import gather_data
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import scienceplots
from src.data_analysis.state_frequency.state_counter import StateCounter

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

def plot_checkers(res=300):
    print('~~~~~~~~~~~~~~~~~~~ Plotting checkers board positions (appendix) ~~~~~~~~~~~~~~~~~~~')
    save_data = False
    if save_data:
        state_counter = gather_data('checkers', [0,1,2,3,4,5,6], max_file_num=10, save_turn_num=True)
        state_counter.prune_low_frequencies(threshold=10)
        with open('../plot_data/board_positions/checkers_counter.pkl', 'wb') as f:
            pickle.dump(state_counter, f)
    print('Loading data')
    with open('../plot_data/board_positions/checkers_counter.pkl', 'rb') as f:
        state_counter: StateCounter = pickle.load(f)
    keys = [i for i,_ in state_counter.frequencies.most_common()]
    early_turns = [0,9,99,999]
    late_turns = [39, 140, 556, 3345]
    for index in early_turns + late_turns:
        plot_checkers_state(keys[index], str(index))

    print('Plotting subplots')

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Figure title')
    names = early_turns + late_turns
    row_titles = ['Checkers Board states', 'Late-game board states']
    # create 3x1 subfigs
    subfigs: list[matplotlib.figure.FigureBase] = fig.subfigures(nrows=2, ncols=1)
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
            ax.set_title(f'State \#{name+1}', fontsize=16)
    plt.tight_layout()
    fig.savefig('./plots/checkers_positions.png', dpi=res)


def parse_oware_state(state_str: str):
    lines = state_str.split('\n')
    # Extract the player scores
    player_1_score = int(lines[0].split('=')[1][:3].strip())
    player_0_score = int(lines[-2].split('=')[1][:3].strip())
    # Extract the board state
    board_state = []
    for row in range(2, 4):
        row_state = [int(x) for x in lines[row].split()]
        board_state.append(row_state)
    # Extract the bank state
    bank_state = [player_1_score, player_0_score]
    return board_state, bank_state


def plot_oware_state(state, name: str):
    # define board dimensions
    board_width = 6
    board_height = 2
    hole_radius = 0.46
    font_size = 24
    bank_height = 0.8
    # define the initial state of the board
    board_state, bank_state = parse_oware_state(str(state))
    # create the board
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='#A67C52')
    ax.set_xlim(0, board_width)
    ax.set_ylim(0, board_height + 2*bank_height)
    ax.set_aspect('equal')
    ax.axis('off')
    # define the color palette
    board_color = 'sienna'
    hole_color = 'peru'
    piece_color = 'white'
    # draw the holes and display the piece counts
    for row in range(board_height):
        for col in range(board_width):
            x = col + 0.5
            y = row + 0.5 + bank_height
            circle = patches.Circle((x, y), hole_radius, fill=True, facecolor=hole_color,linewidth=2, edgecolor='black')
            ax.add_artist(circle)
            piece_count = board_state[row][col]
            ax.text(x, y, str(piece_count), ha='center', va='center', fontsize=font_size, color=piece_color)
    # draw the bank holes and display the captured piece counts
    def bank(side: str):
        # create bank holes using Matplotlib patches
        if side == 'top':
            rec_pos = (0, board_height + bank_height)
            state = 0
        elif side == 'bottom':
            rec_pos = (0, 0)
            state = 1
        else:
            raise ValueError('side must be either "top" or "bottom", got ' + side)
        bank = patches.Rectangle(rec_pos, board_width, bank_height, fill=True, facecolor=board_color, linewidth=4, edgecolor='black')
        bank.set_capstyle('round')
        ax.add_artist(bank)
        x, y = np.array(rec_pos) + np.array([board_width / 2, bank_height / 2])
        ax.text(x, y, 'Score: ' + str(bank_state[state]), ha='center', va='center', fontsize=font_size + 4, color=piece_color)
    bank('top')
    bank('bottom')

    fig.savefig(f'plots/board_positions/oware-board_{name}.png', dpi=300, bbox_inches='tight', pad_inches=0)


def plot_oware(res=300):
    print('~~~~~~~~~~~~~~~~~~~ Plotting Oware board positions (appendix) ~~~~~~~~~~~~~~~~~~~')
    save_data = False
    if save_data:
        state_counter = gather_data('oware', [0,1,2,3,4,5], max_file_num=10, save_turn_num=True)
        state_counter.prune_low_frequencies(threshold=10)
        with open('../plot_data/board_positions/oware_counter.pkl', 'wb') as f:
            pickle.dump(state_counter, f)
    print('Loading data')
    with open('../plot_data/board_positions/oware_counter.pkl', 'rb') as f:
        state_counter: StateCounter = pickle.load(f)
    keys = [i for i,_ in state_counter.frequencies.most_common()]
    early_turns = [0,9,99,1004]
    late_turns = [163, 334, 1419, 11648]
    for index in early_turns + late_turns:
        plot_oware_state(keys[index], str(index))

    print('Plotting subplots')

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Figure title')
    names = early_turns + late_turns
    row_titles = ['Oware Board states', 'Late-game board states']
    # create 3x1 subfigs
    subfigs: list[matplotlib.figure.FigureBase] = fig.subfigures(nrows=2, ncols=1)
    png_dir = 'plots/board_positions/'
    # Iterate over each name
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(row_titles[row], fontsize=18)
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=4)
        for col, ax in enumerate(axs):
            name = names[row*4+col]
            png_file = os.path.join(png_dir, f'oware-board_{name}.png')
            img = plt.imread(png_file)
            ax.imshow(img)
            ax.axis('off') 
            ax.set_title(f'State \#{name+1}', fontsize=16)
    plt.tight_layout()
    fig.savefig('./plots/oware_positions.png', dpi=res)