import torch
import numpy as np
import chess


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Takes the board state and converts it to a one-hot tensor.
    """
    # An 8x8 board with a third dimension of size 12 to separate the pieces into different planes.
    np_board = np.zeros((12, 8, 8))

    # Gives a value to each of the types of pieces.
    piece_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
                 "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11}
    
    # Inserts a 1 in each of the pieces' planes where they are located.
    for square, piece in board.piece_map().items():
        plane = piece_map[str(piece)]
        row, col = divmod(square, 8)
        np_board[plane, row, col] = 1
    
    # Returns a tensor of the one-hot
    return torch.tensor(np_board, dtype=torch.float32)


if __name__ == "__main__":
    board = chess.Board()

    print(board_to_tensor(board))