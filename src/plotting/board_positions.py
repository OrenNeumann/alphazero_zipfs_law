from cairosvg import svg2png
import chess.svg

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
            if board[row][col] == '.':
                empty += 1
            elif board[row][col] == 'o':
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += 'P'
            elif board[row][col] == '+':
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += 'p'
            else:
                print(row,col)
                raise ValueError('unrecognized piece: \"' + board[row][col]+'\"')
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    fen = '/'.join(fen_rows)
    return f"{fen} w - - 0 1"


def plot_checkers_state(state):
    fen = checkers_to_fen(str(state))
    board = chess.Board(fen)
    svg_text = chess.svg.board(
        board,
        size=350)
    svg2png(bytestring=svg_text, write_to='plots/checkers-board.png')