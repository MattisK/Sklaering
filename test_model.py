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
from stats import get_confidence_interval_and_average
import json
import os

# TODO: shorten worst fish times
# TODO: add comments
# TODO: add a variable to turn off saving to json, and just print the results

def games_to_json(results: dict) -> None:
    if os.path.exists("results.json"):
        with open("results.json", "r") as file:
            json_dict = json.load(file)
        json_dict[f"version_{len(json_dict) + 1}"] = results
    else:
        json_dict = {"version_1": results}

    json_object = json.dumps(json_dict, indent=4)

    with open("results.json", "w") as outfile:
        outfile.write(json_object)

def play_game_stockfish(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> tuple[str, int, list]:
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

def get_worst_move(board: chess.Board) -> chess.Move:
    """Returns the worst move according to Stockfish."""
    stockfish.set_fen_position(board.fen())
    legal_moves = list(board.legal_moves)
    worst_move = None
    worst_eval = float('inf') if board.turn == chess.WHITE else float('-inf')

    for move in legal_moves:
        board.push(move)
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation().get('value', 0)
        # Black wants a higher evaluation
        if evaluation > worst_eval:
            worst_eval = evaluation
            worst_move = move
        board.pop()

    # Fallback: return the first legal move if no "worst" move is found
    if worst_move is None:
        worst_move = legal_moves[0]

    return worst_move

def play_game_worstfish(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> tuple[str, int, list]:
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
            worst_move = get_worst_move(board)
            if worst_move:
                board.push(worst_move)

        length += 1

    return board.result(), length, ai_move_times

def get_random_move(board: chess.Board) -> chess.Move:
    """Returns a random legal move."""
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)

def play_game_random_move(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> tuple[str, int, list]:
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
    #stockfish_path = "C:/Users/Matti/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2"
    stockfish_path = "C:/Users/chris/OneDrive/Desktop/stockfish/stockfish/stockfish-windows-x86-64-avx2"
    #stockfish_path = "C:/#DTU/3 ugers dec2025/Sklaering/stockfish/stockfish-windows-x86-64-avx2.exe"
    stockfish = Stockfish(stockfish_path, depth=1)
    stockfish.set_skill_level(0)

    # Keeps track of the results.
    results_stock = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    results_stock_1 = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    results_stock_2 = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    results_worst = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    results_random = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    num_games = 10

    for type in ["stock", "stock_1", "stock_2", "worst", "random"]:
        for i in range(num_games):
            print(f"Game: {i + 1} ({type})")
            board = chess.Board()  # Reset the board for each game
            if type == "stock":
                result, length, ai_move_times = play_game_stockfish(model, board, stockfish)
                results_stock["MoveCounts"].append(length)
                results_stock["AIMoveTimes"].extend(ai_move_times)
                if result == "1-0":
                    results_stock["White"] += 1
                elif result == "0-1":
                    results_stock["Black"] += 1
                elif result == "1/2-1/2":
                    results_stock["Draw"] += 1
            elif type == "stock_1":
                stockfish = Stockfish(stockfish_path, depth=2)
                result, length, ai_move_times = play_game_stockfish(model, board, stockfish)
                results_stock_1["MoveCounts"].append(length)
                results_stock_1["AIMoveTimes"].extend(ai_move_times)
                if result == "1-0":
                    results_stock_1["White"] += 1
                elif result == "0-1":
                    results_stock_1["Black"] += 1
                elif result == "1/2-1/2":
                    results_stock_1["Draw"] += 1
            elif type == "stock_2":
                stockfish = Stockfish(stockfish_path, depth=4)
                result, length, ai_move_times = play_game_stockfish(model, board, stockfish)
                results_stock_2["MoveCounts"].append(length)
                results_stock_2["AIMoveTimes"].extend(ai_move_times)
                if result == "1-0":
                    results_stock_2["White"] += 1
                elif result == "0-1":
                    results_stock_2["Black"] += 1
                elif result == "1/2-1/2":
                    results_stock_2["Draw"] += 1
            elif type == "worst":
                result, length, ai_move_times = play_game_worstfish(model, board, stockfish)
                results_worst["MoveCounts"].append(length)
                results_worst["AIMoveTimes"].extend(ai_move_times)
                if result == "1-0":
                    results_worst["White"] += 1
                elif result == "0-1":
                    results_worst["Black"] += 1
                elif result == "1/2-1/2":
                    results_worst["Draw"] += 1
            elif type == "random":
                result, length, ai_move_times = play_game_random_move(model, board, stockfish)
                results_random["MoveCounts"].append(length)
                results_random["AIMoveTimes"].extend(ai_move_times)
                if result == "1-0":
                    results_random["White"] += 1
                elif result == "0-1":
                    results_random["Black"] += 1
                elif result == "1/2-1/2":
                    results_random["Draw"] += 1
        if type == "stock":
            results_stock["AIMoveTimes"] = get_confidence_interval_and_average(results_stock["AIMoveTimes"])
        elif type == "stock_1":
            results_stock_1["AIMoveTimes"] = get_confidence_interval_and_average(results_stock_1["AIMoveTimes"])
        elif type == "stock_2":
            results_stock_2["AIMoveTimes"] = get_confidence_interval_and_average(results_stock_2["AIMoveTimes"])
        elif type == "worst":
            results_worst["AIMoveTimes"] = get_confidence_interval_and_average(results_worst["AIMoveTimes"])
        elif type == "random":
            results_random["AIMoveTimes"] = get_confidence_interval_and_average(results_random["AIMoveTimes"])
        
    # Save game results to file
    result_to_json = {"NumGames": num_games, "Stockfish": results_stock, "Stockfish 1": results_stock_1, "Stockfish 2":results_stock_2, "Worstfish": results_worst, "Random": results_random}

    games_to_json(result_to_json)
