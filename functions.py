import torch
import numpy as np
import chess


def board_to_tensor(board: chess.Board):
    np_board = np.zeros((12, 8, 8))
    piece_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
                 "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11}
    
    for square, piece in board.piece_map().items():
        plane = piece_map[str(piece)]
        row, col = divmod(square, 8)
        np_board[plane, row, col] = 1
    
    return torch.tensor(np_board, dtype=torch.float32)


board = chess.Board()

print(board_to_tensor(board))