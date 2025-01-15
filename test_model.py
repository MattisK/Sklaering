import time
import chess
import datetime
from ChessCNN import ChessCNN
from functions import get_best_move
from stockfish import Stockfish
import torch
import pandas as pd
from scipy.stats import ttest_ind, t
import numpy as np
import random
import chess
import time

# TODO: shorten ai times
# TODO: add comments
# TODO: add more tests fx random , worstfish  etc.




def save_game(results: dict, path: str) -> None:
    """Gemmer spilresultaterne for alle modstandertyper i en tekstfil."""
    x = datetime.datetime.now()
    with open(path, "a") as f:
        f.write(f"\nResults for {x}:")
        for opponent_type, opponent_results in results.items():
            f.write(f"\n--- Against {opponent_type} ---")
            f.write(f"\nWhite wins: {opponent_results['White']}")
            f.write(f"\nDraws: {opponent_results['Draw']}")
            f.write(f"\nMove Counts: {opponent_results.get('MoveCounts', [])}")
            f.write(f"\nAI Move Times: {opponent_results.get('AIMoveTimes', [])}")
        f.write("\n===================================\n")

def play_game_stockfish(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> (str, int, list):
    """Plays a game and returns the result, number of moves, and list of AI move times."""
    length = 0
    ai_move_times = []  # List to store time taken by AI for each move

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # White (AI model) turn
            start_time = time.time()  # Start timing the AI's move
            move = get_best_move(model, board)
            board.push(move)
            stockfish.set_fen_position(board.fen())
            end_time = time.time()  # End timing the AI's move
            ai_move_times.append(end_time - start_time)  # Record AI move time
        else:
            # Black (Stockfish) turn
            stockfish_best_move = stockfish.get_best_move()
            stockfish_move = chess.Move.from_uci(stockfish_best_move)
            if stockfish_move:
                board.push(stockfish_move)

        length += 1

    return board.result(), length, ai_move_times
import chess
import time
from stockfish import Stockfish

def get_worst_move(stockfish: Stockfish, board: chess.Board) -> chess.Move:
    """Returns the worst move according to Stockfish."""
    stockfish.set_fen_position(board.fen())
    legal_moves = list(board.legal_moves)
    worst_move = None
    worst_eval = float('inf') if board.turn == chess.WHITE else float('-inf')

    for move in legal_moves:
        board.push(move)
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation()['value']
        if board.turn == chess.WHITE:
            # White wants a lower evaluation
            if evaluation < worst_eval:
                worst_eval = evaluation
                worst_move = move
        else:
            # Black wants a higher evaluation
            if evaluation > worst_eval:
                worst_eval = evaluation
                worst_move = move
        board.pop()

    return worst_move

def play_game_worstfish(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> (str, int, list):
    """Plays a game and returns the result, number of moves, and list of AI move times."""
    length = 0
    ai_move_times = []  # List to store time taken by AI for each move

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # White (AI model) turn
            start_time = time.time()  # Start timing the AI's move
            move = get_best_move(model, board)
            board.push(move)
            stockfish.set_fen_position(board.fen())
            end_time = time.time()  # End timing the AI's move
            ai_move_times.append(end_time - start_time)  # Record AI move time
        else:
            # Black (Worst move according to Stockfish) turn
            worst_move = get_worst_move(stockfish, board)
            if worst_move:
                board.push(worst_move)

        length += 1

    return board.result(), length, ai_move_times

def get_random_move(board: chess.Board) -> chess.Move:
    """Returns a random legal move."""
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)

def play_game_random_move(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> (str, int, list):
    """Plays a game where one side always makes a random move and returns the result, number of moves, and list of AI move times."""
    length = 0
    ai_move_times = []  # List to store time taken by AI for each move

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # White (AI model) turn
            start_time = time.time()  # Start timing the AI's move
            move = get_best_move(model, board)
            board.push(move)
            stockfish.set_fen_position(board.fen())
            end_time = time.time()  # End timing the AI's move
            ai_move_times.append(end_time - start_time)  # Record AI move time
        else:
            # Black (Random move) turn
            random_move = get_random_move(board)
            board.push(random_move)

        length += 1

    return board.result(), length, ai_move_times

if __name__ == "__main__":
    compare = True
    # Load the trained model.
    model = ChessCNN()
    model.load_state_dict(torch.load("chess_model.pth"))
    model.eval()

    # Initialize stockfish and set the skill level.
    stockfish_path = "C:/Users/Mattis/Desktop/Stockfish/stockfish/stockfish-windows-x86-64-avx2"
    stockfish = Stockfish(stockfish_path, depth=1)
    stockfish.set_skill_level(0)

    # Keeps track of the results.
    results_stock = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    results_worst = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    results_random = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    num_games = 10

    for type in ["stock", "worst", "random"]:
        for i in range(num_games):
            print(f"Game: {i + 1} ({type})")
            board = chess.Board()  # Reset the board for each game
            if type == "stock":
                result, length, ai_move_times = play_game_stockfish(model, board, stockfish)
                results_stock["MoveCounts"].append(length)
                results_stock["AIMoveTimes"].append(ai_move_times)
                if result == "1-0":
                    results_stock["White"] += 1
                elif result == "0-1":
                    results_stock["Black"] += 1
                elif result == "1/2-1/2":
                    results_stock["Draw"] += 1
            elif type == "worst":
                result, length, ai_move_times = play_game_worstfish(model, board, stockfish)
                results_worst["MoveCounts"].append(length)
                results_worst["AIMoveTimes"].append(ai_move_times)
                if result == "1-0":
                    results_worst["White"] += 1
                elif result == "0-1":
                    results_worst["Black"] += 1
                elif result == "1/2-1/2":
                    results_worst["Draw"] += 1
            elif type == "random":
                result, length, ai_move_times = play_game_random_move(model, board, stockfish)
                results_random["MoveCounts"].append(length)
                results_random["AIMoveTimes"].append(ai_move_times)
                if result == "1-0":
                    results_random["White"] += 1
                elif result == "0-1":
                    results_random["Black"] += 1
                elif result == "1/2-1/2":
                    results_random["Draw"] += 1

    # Save game results to file
    save_game({"Stockfish": results_stock, "Worstfish": results_worst, "Random": results_random}, "save_file.txt")
