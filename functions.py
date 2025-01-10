import torch
import numpy as np
import chess
import chess.pgn
from stockfish import Stockfish
from ChessModel import ChessModel


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

    # Create a dictionary with moves as keys and their indices as values
    move_idx = {move: idx for idx, move in enumerate(sorted(all_moves))}

    return move_idx


def encode_move(move: chess.Move, move_idx: dict) -> int:
    """
    Returns the index in the move_idx dictionary for a given move, so each move has a unique index.
    """
    return move_idx[move.uci()]


def get_model_move(board: chess.Board, model: ChessModel) -> chess.Move:
    """
    Gets all legal moves for a position and returns the move with the highest predicted value from the model.
    """
    # Board tensor.
    input_tensor = board_to_tensor(board).unsqueeze(0) # gets tensor with the dimensions 1x12x8x8
    

    # find all legal moves for a board state
    legal_moves = [move.uci() for move in board.legal_moves] # gets all legal moves for the board
    
    max_move = legal_moves[0]  # sets the best move to None
    temp_board = board.copy() # copy the board
    temp_board.push(chess.Move.from_uci(legal_moves[0])) # pushes the first move to the board
    max_value = model(board_to_tensor(temp_board).unsqueeze(0)) # sets the value to the first move
    print(float(max_value.squeeze(0)))
    
    for move in legal_moves: # for each move in legal moves 
        
        # get the board state of the move
        temp = board.copy()
        move_to_board = chess.Move.from_uci(move)
        temp.push(move_to_board) 
        print(move, ":", float(model(board_to_tensor(temp).unsqueeze(0))))
        
        # if the value of the move is better than the previous move
        if float(model(board_to_tensor(temp).unsqueeze(0))) > float(max_value):
            max_move = move # sets the move to the best move
            max_value = float(model(board_to_tensor(temp).unsqueeze(0))) # sets the value to the best value
            
    print(max_move)
    return chess.Move.from_uci(max_move)
    
    """    
    # Disables gradient computation for memory efficiency and shorter duration.
    with torch.no_grad(): #TODO: maybe do it with gradient so we can see if we overfit
        # Feed the board tensor to the model (predicted moves).
        outputs = model(input_tensor)

        top_k = torch.topk(outputs, k=1, dim=1) #TODO: review this to make sure it doesnt make any random moves
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
                bool_list.append(False)
                return predicted_move
            else:
                bool_list.append(True)
        """

def reward(board: chess.Board, move, method: str) -> float:
    """
    Function for evaluating the reward of a given board state.
    Method: material, check, stockfish or human.
    """
    #convert move to board
    temp_board = board.copy()
    temp_board.push(move)
    if method == "material":
        return reward_material(board, temp_board)
    elif method == "check":
        return reward_check(board, temp_board)
    elif method == "stockfish":
        return reward_stockfish(board, temp_board)
    elif method == "human":
        return reward_human(board, temp_board)
    else:
        raise ValueError("Invalid method. Choose between 'material', 'check', 'stockfish', or 'human'.")

def reward_material(board: chess.Board, next_board: chess.Board) -> float:
    """
    Function for evaluating the reward of a given board state based on material.
    """
    def material_value(board: chess.Board) -> int:
        material_score = 0
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0  # King's value is not counted for material evaluation
        }
        for piece_type in piece_values:
            material_score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            material_score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        return material_score

    return material_value(next_board) - material_value(board)

def reward_check(board: chess.Board, next_board: chess.Board) -> float:
    """
    Function for evaluating the reward of a given board state based on check and checkmate.
    """
    if next_board.is_checkmate():
        return 10
    elif next_board.is_check():
        return 1
    else:
        return 0

def reward_stockfish(board: chess.Board, next_board: chess.Board) -> float:
    """
    Function for evaluating the reward of a given board state based on Stockfish evaluation.
    """
    # convert the nextboard to a stockfish board
    stockfish_path = "C:/Users/Mattis/Desktop/Stockfish/stockfish/stockfish-windows-x86-64-avx2"
    "C:/Users/Mattis/Desktop/Stockfish/stockfish/stockfish-windows-x86-64-avx2"

    stockfish = Stockfish(path=stockfish_path, depth=15)
    stockfish.set_fen_position(fen_position=next_board.fen())
    eval = stockfish.get_evaluation()

    if eval["type"] == "cp":  # Centipawn score the amount of pawns the engine thinks it is ahead
        if eval["value"] > 0:
            return 1
        elif eval["value"] < 0:
            return -1
        else:
            return 0

    elif eval["type"] == "mate":  # Mate in N moves
        # Assign a very high positive/negative reward for mate
        if eval["value"] > 0:  # Positive means White is winning
            return 10
        else:  # Negative means Black is winning
            return -10
    else:
        # Default fallback in case of unexpected evaluation type
        return 0.0


def reward_human(board: chess.Board, next_board: chess.Board) -> float:
    """
    Function for evaluating the reward of a given board state based on human evaluation.
    Assume that there is a chess board for the given board state.
    """
    try:
        moves = np.load("moves.npy", allow_pickle=True)  # read data
    except FileNotFoundError:
        print("Error: moves.npy file not found.")
        return 0.0

    next_fen = board.fen()  # FEN for the next board
    # find board for the move that was made and check if that board is the same as the next board
    for move_data in moves:
        pass

    return 0.0  # Default, hvis ingen match findes

