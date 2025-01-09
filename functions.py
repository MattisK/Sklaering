import torch
import numpy as np
import chess
import chess.pgn
import random

from jinja2.lexer import TOKEN_DOT

from ChessModel import ChessModel
import time
import os


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


def parse_pgn(pgn_file_path: str, num_games: int) -> list:
    """
    Function for extracting the games from a PGN file from the Lichess database and make a list with a board-move pair.
    """
    moves = []# TODO: memory optimise this and maybe, add a try-except block to catch errors
    counter = 0

    with open(pgn_file_path, "r") as pgn_file:
        # Loop runs while the read game is not None.
        while counter < num_games:
            # Load the game.
            game = chess.pgn.read_game(pgn_file)

            # Break loop if game is None.
            if game is None:
                break
            
            if game.headers.get("Termination") == "Normal" and game.headers.get("Result") == "1-0":
                # The chess board.
                board = chess.Board()

                # Appends each move and the current board in the game as a tuple to the moves list. Also pushes the move to the board.
                for move in game.mainline_moves():
                    moves.append((board.copy(), move))
                    board.push(move)

                counter += 1
                print("Iteration:", f"{counter}/{num_games}")

    print(f"Done generating moves.")
    return moves
    

def move_to_idx(board: chess.Board) -> dict:
    """
    Takes all the possible moves and promotions and returns a dictionary with the moves as keys and their index as values.
    """
    # Generate all legal moves from the current board state
    all_moves = [move.uci() for move in board.generate_legal_moves()]
    #TODO: check if this is where the 'check' error is
    print(all_moves)

    # Create a dictionary with moves as keys and their indices as values
    move_idx = {move: idx for idx, move in enumerate(sorted(all_moves))}

    return move_idx


def encode_move(move: chess.Move, move_idx: dict) -> int:
    """
    Returns the index in the move_idx dictionary for a given move, so each move has a unique index.
    """
    return move_idx[move.uci()]


def get_model_move(board: chess.Board, model: ChessModel) -> tuple[chess.Move, int]:
    """
    Gets the move for the model.
    """
    # Board tensor.
    input_tensor = board_to_tensor(board).unsqueeze(0)

    # Disables gradient computation for memory efficiency and shorter duration.
    with torch.no_grad(): #TODO: maybe do it with gradient so we can see if we overfit
        # Feed the board tensor to the model (predicted moves).
        outputs = model(input_tensor)

        top_k = torch.topk(outputs, k=5, dim=1) #TODO: review this to make sure it doesnt make any random moves
        predicted_indices = top_k.indices[0].tolist() #TODO: understand top_k better

    # Get the index, move dictionary. (by switching the keys and values from move_idx)
    move_idx = move_to_idx(board)
    idx_to_move = {idx: move for move, idx in move_idx.items()}

    bool_list = []

    for predicted_idx in predicted_indices:
        if predicted_idx in idx_to_move:
            predicted_move_uci = idx_to_move[predicted_idx]
            predicted_move = chess.Move.from_uci(predicted_move_uci)

            # Return predicted move if it is legal and make the play on the board.
            if predicted_move in board.legal_moves:
                return predicted_move, 0
            else:
                bool_list.append(True)
        else:
            if all(bool_list): # TODO: remake this (random moves)
                global random_counter
                random_counter = 1
                random_move = random.choice(list(idx_to_move.keys()))
                random_move_uci = idx_to_move[random_move]
                random_move_final = chess.Move.from_uci(random_move_uci)
                return random_move_final, random_counter