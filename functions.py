import torch
import numpy as np
import chess
import chess.pgn
from collections import defaultdict


def board_to_tensor(board: chess.Board) -> torch.Tensor:
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
    
    # Returns a tensor of the one-hot matrix
    return torch.tensor(np_board, dtype=torch.float32)


def parse_pgn(pgn_file_path: str, num_games: int) -> None:
    """
    Function for extracting the games from a PGN file from the Lichess database and make a list with a board-move pair.
    """
    moves = []
    counter = 0
    
    with open(pgn_file_path, "r") as pgn_file:
        # Loop runs while the read game is not None.
        while counter < num_games:
            # Load the game.
            game = chess.pgn.read_game(pgn_file)

            # Break loop if game is None
            if game is None:
                break
            
            # The chess board.
            board = chess.Board()

            # Appends each move and the current board in the game as a tuple to the moves list. Also pushes the move to the board.
            for move in game.mainline_moves():
                moves.append((board.copy(), move))
                board.push(move)

            counter += 1
            
            print("Iteration:", counter)

    print("Done generating moves.")
    return moves
    

def move_to_idx(board: chess.Board) -> dict:
    """Takes all the possible moves and promotions and returns a dictionary with the moves as keys and their index as values."""
    # Calculate the all the possible moves and promotions
    all_moves = []

    # The square the piece is moving from.
    for square_from in chess.SQUARES:
        # The square the piece is moving to.
        for square_to in chess.SQUARES:
            # Check for promotion.
            for promotion in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                try:
                    # Construct the move and check if it is legal. If so, append the uci value to all_moves.
                    move = chess.Move(square_from, square_to, promotion=promotion)
                    if board.is_legal(move):
                        all_moves.append(move.uci())
                except:
                    # Exception for the case where move could not be constructed with the current parameters.
                    continue

    # Remove duplicate moves and sort the list.
    all_moves = sorted(set(all_moves))

    # Make a dictionary with the moves as keys and their indices as values.
    move_idx = {move: idx for idx, move in enumerate(all_moves)}

    return move_idx


def encode_move(move: chess.Move, move_idx: dict) -> int:
    """Returns the index in the move_idx dictionary for a given move, so each move has a unique index."""
    return move_idx[move.uci()]
