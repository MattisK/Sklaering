import chess
import numpy as np
import torch
from ChessCNN import ChessCNN


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Takes the board state and converts it to a 12x8x8 one-hot encoded tensor.
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
    
    # Returns a tensor of the one-hot encoded tensor
    return torch.tensor(np_board, dtype=torch.float32)


def encode_move(move: chess.Move) -> int:
    """
    Converts the move to an integer and returns this. It is unique for the given move.
    """
    return move.from_square * 64 + move.to_square


def get_best_move(model: ChessCNN, board: chess.Board) -> chess.Move:
    """
    Looks at the model output policy and returns the legal move with the highest probability.
    """
    # Convert the board to a tensor and unsqueeze it.
    board_tensor = encode_board(board).unsqueeze(0)

    # Set the model to evaluation mode.
    model.eval()

    # Disable gradient computation and get the policy from the model with the input board tensor.
    with torch.no_grad():
        policy, _ = model(board_tensor)
    
    # Make a list of legal moves.
    legal_moves = list(board.legal_moves)

    # A probability tensor filled with zeros of size legal_moves. Will be filled with probabilities.
    move_probabilities = torch.zeros(len(legal_moves))
    
    # Enumerates the legal moves and gets the encoded move as index in the policy
    # from the model and adds this to the probability tensor.
    for i, move in enumerate(legal_moves):
        move_idx = encode_move(move)
        move_probabilities[i] = policy[0, move_idx]
    
    # The index of the best move is found by the argmax method from the torch library.
    best_move_idx = torch.argmax(move_probabilities).item()

    # Returns the move in the list of legal moves with the index of the move
    # with the highest probability.
    return legal_moves[best_move_idx]
