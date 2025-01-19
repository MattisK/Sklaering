import chess
from ChessCNN import ChessCNN
from functions import get_best_move
from stockfish import Stockfish
import torch


def play_game(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> None:
    """While the game is not over, plays the game against the stockfish chess engine."""
    # Checks if the game is over.
    while not board.is_game_over():
        # White (the model) turn.
        if board.turn == chess.WHITE:
            # Gets the best move according to the model and pushes this to the board.
            # Then sets the fen position for the stockfish engine.
            move = get_best_move(model, board)
            board.push(move)
            stockfish.set_fen_position(board.fen())
        else:
            # Black (stockfish) turn.
            stockfish_best_move = stockfish.get_best_move()
            stockfish_move = chess.Move.from_uci(stockfish_best_move)
            if stockfish_move:
                board.push(stockfish_move)


if __name__ == "__main__":
    # Load the trained model.
    model = ChessCNN()
    model.load_state_dict(torch.load("chess_model_early_stopping.pth"))

    # Set the model to evaluation mode.
    model.eval()

    # Initialize stockfish and set the skill level.
    #stockfish_path = "C:/Users/chris/Desktop/Stockfish/stockfish/stockfish-windows-x86-64-avx2"
    stockfish_path = "C:/Users/chris/OneDrive/Desktop/stockfish/stockfish/stockfish-windows-x86-64-avx2"
    stockfish = Stockfish(stockfish_path, depth=1)
    stockfish.set_skill_level(0)

    # Board object.
    board = chess.Board()
    
    # Keeps track of the results.
    results = {"White": 0, "Black": 0, "Draw": 0}
    num_games = 1000
    for i in range(num_games):
        # Runs loop for each game and appends the result to the result dictionary,
        # then resets the board when the game is over and prints the result when all games are over.
        play_game(model, board, stockfish)
        print("Game:", i + 1)
        result = board.result()
        if result == "1-0":
            results["White"] += 1
        elif result == "0-1":
            results["Black"] += 1
        elif result == "1/2-1/2":
            results["Draw"] += 1

        board.reset()

print(results)