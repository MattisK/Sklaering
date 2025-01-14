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
    # The encode_board function converts a chess board state into a 12x8x8 one-hot encoded tensor using the PyTorch library.
    # This creates an empty board representation where each of the 12 planes corresponds to a different type
    # of chess piece (6 for white pieces and 6 for black pieces).

    # This dictionary maps each type of piece to an index. Lowercase letters represent black pieces, and uppercase letters represent white pieces.
    piece_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
                "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11}
    
    # This loop iterates over each piece on the board, determines its position, and sets the corresponding position in the NumPy array to 1.
    # The divmod function is used to convert the square index into row and column indices.
    for square, piece in board.piece_map().items():
        plane = piece_map[str(piece)]
        row, col = divmod(square, 8)
        np_board[plane, row, col] = 1
    
    # Returns a tensor of the one-hot encoded tensor
    # Finally, the function converts the NumPy array to a PyTorch tensor with a data type of float32 and returns it.
    return torch.tensor(np_board, dtype=torch.float32)


def encode_move(move: chess.Move) -> int:
    """
    Converts the move to an integer and returns this. It is unique for the given move.
    """
    # move.from_square: This attribute of the chess.Move object represents the starting square of the move.
    # It is an integer between 0 and 63, where each number corresponds to a specific square on the chessboard.
    # The function calculates a unique integer by multiplying move.from_square by 64 and then adding move.to_square
    # This ensures that each combination of from_square and to_square results in a unique integer because the chessboard has 64 squares,
    # and multiplying by 64 shifts the from_square value to a unique range.
    return move.from_square * 64 + move.to_square


def get_best_move(model: ChessCNN, board: chess.Board) -> chess.Move:
    """
    Looks at the model output policy and returns the legal move with the highest probability.
    """
    # The function takes two parameters: model (an instance of ChessCNN) and board (an instance of chess.Board).
    # It returns a chess.Move object, which represents the best move according to the model.

    # Convert the board to a tensor and unsqueeze it.
    # The encode_board function converts the chess board state into a tensor. The unsqueeze method adds a dimension to the tensor
    # to match the expected input shape of the model.
    board_tensor = encode_board(board).unsqueeze(0)

    # Set the model to evaluation mode.
    # The model is set to evaluation mode using model.eval(). This disables certain layers like dropout, which are only used during training.
    model.eval()

    # Disable gradient computation and get the policy from the model with the input board tensor.
    # outputs a policy tensor, which contains probabilities for each possible move.

    with torch.no_grad():
        policy, _ = model(board_tensor)
    
    # legal_moves is a list of all legal moves in the current board state.
    legal_moves = list(board.legal_moves)

    # A probability tensor filled with zeros of size legal_moves. Will be filled with probabilities.
    # move_probabilities is a tensor initialized with zeros, with a size equal to the number of legal moves.
    move_probabilities = torch.zeros(len(legal_moves))
    
    # Enumerates the legal moves and gets the encoded move as index in the policy
    # from the model and adds this to the probability tensor.
    for i, move in enumerate(legal_moves):
        move_idx = encode_move(move)
        move_probabilities[i] = policy[0, move_idx]
    
    # torch.argmax finds the index of the highest probability in move_probabilities.
    # .item() converts the resulting tensor to a Python integer.
    best_move_idx = torch.argmax(move_probabilities).item()

    # Returns the move in the list of legal moves with the index of the move
    # with the highest probability.
    return legal_moves[best_move_idx]
