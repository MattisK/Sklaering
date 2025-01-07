import chess
import numpy as np


def reset_board():  # resets board to starting position
    """
    Resets the board to the starting position.
    """
    board = chess.Board()
    return board


def chess_to_matrix(move):
    """
    Converts a move in chess notation to a 2-tuple of row and column indices.
    """
    row = ord(move[0]) - ord('a')
    column = 8 - int(move[1])
    return row, column


def apply_move(move, board):
    """
    Applies a SAN move to the board and converts it to an 8x8x12 one-hot encoded matrix.
    """
    board.push_san(move)  # Apply the move to the python-chess board

    # Convert to 8x8x12 one-hot encoded matrix
    one_hot_board = board_to_one_hot(board)

    return one_hot_board


def board_to_one_hot(board: chess.Board):
    np_board = np.zeros((12, 8, 8))
    piece_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
                 "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11}

    for square, piece in board.piece_map().items():
        plane = piece_map[str(piece)]
        row, col = divmod(square, 8)
        np_board[plane, row, col] = 1

    return np_board