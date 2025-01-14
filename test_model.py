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

# TODO: shorten ai times
# TODO: add comments
# TODO: add more tests fx random , worstfish  etc.


def compare_versions_with_ttests(file_path: str):
    """Sammenligner versioner i filen parvist og udskriver forskelle, t-tests og 95% konfidensintervaller."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find blokke baseret på "Results for" linjer
    blocks = []
    current_block = []
    for line in lines:
        if line.startswith("Results for"):
            if current_block:
                blocks.append(current_block)
            current_block = [line.strip()]
        elif line.strip():
            current_block.append(line.strip())
    if current_block:
        blocks.append(current_block)

    if len(blocks) < 2:
        print("Der er ikke nok resultater til at lave en sammenligning.")
        return

    # Helper-funktion til at parse resultater
    def parse_block(block):
        results = {"White wins": 0, "Black wins": 0, "Draws": 0, "Move Counts": [], "AI Move Times": []}
        for line in block:
            if line.startswith("White wins"):
                results["White wins"] = int(line.split(": ")[1])
            elif line.startswith("Black wins"):
                results["Black wins"] = int(line.split(": ")[1])
            elif line.startswith("Draws"):
                results["Draws"] = int(line.split(": ")[1])
            elif line.startswith("Move Counts"):
                results["Move Counts"] = list(map(int, line.split(": ")[1][1:-1].split(", ")))
            elif line.startswith("AI Move Times"):
                try:
                    ai_times_str = line.split(": ")[1][1:-1]  # Fjern yderste "[" og "]"
                    ai_times_str = ai_times_str.replace("[", "").replace("]", "")  # Fjern interne "[" og "]"
                    ai_times = [[float(x) for x in game.split(", ")] for game in ai_times_str.split("], [")]
                    results["AI Move Times"] = ai_times
                except (IndexError, ValueError):
                    results["AI Move Times"] = []
        return results

    # Beregn 95% konfidensinterval
    def calculate_confidence_interval(data1, data2):
        if len(data1) < 2 or len(data2) < 2:
            return 0, (0, 0)
        diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt(np.var(data1, ddof=1) / len(data1) + np.var(data2, ddof=1) / len(data2))
        margin_of_error = t.ppf(0.975, df=min(len(data1), len(data2)) - 1) * pooled_std
        return diff, (diff - margin_of_error, diff + margin_of_error)

    # Sammenlign alle versioner parvist
    for i in range(len(blocks) - 1):
        version_x = f"Version {i + 1}"
        version_y = f"Version {i + 2}"

        # Parse resultater
        results_x = parse_block(blocks[i])
        results_y = parse_block(blocks[i + 1])

        # Flade AI Move Times
        ai_times_x = [time for game in results_x.get("AI Move Times", []) for time in game]
        ai_times_y = [time for game in results_y.get("AI Move Times", []) for time in game]

        # Beregn forskelle og konfidensintervaller
        white_wins_diff, white_wins_ci = calculate_confidence_interval(
            [results_x["White wins"]], [results_y["White wins"]]
        )
        draws_diff, draws_ci = calculate_confidence_interval(
            [results_x["Draws"]], [results_y["Draws"]]
        )
        moves_diff, moves_ci = calculate_confidence_interval(
            results_x["Move Counts"], results_y["Move Counts"]
        )
        ai_time_diff, ai_time_ci = calculate_confidence_interval(ai_times_x, ai_times_y)

        # T-tests
        ttest_moves = ttest_ind(results_x["Move Counts"], results_y["Move Counts"], equal_var=False)
        ttest_ai_times = ttest_ind(ai_times_x, ai_times_y, equal_var=False)

        # Handle NaN in confidence intervals
        moves_ci = moves_ci if not any(np.isnan(moves_ci)) else (0, 0)
        ai_time_ci = ai_time_ci if not any(np.isnan(ai_time_ci)) else (0, 0)

        # Print rapport
        print(f"\n=== {version_x} vs. {version_y} ===")
        print(f"Forskel i White wins: {white_wins_diff:.2f} (95% CI: {white_wins_ci[0]:.2f}, {white_wins_ci[1]:.2f})")
        print(f"Forskel i remisser: {draws_diff:.2f} (95% CI: {draws_ci[0]:.2f}, {draws_ci[1]:.2f})")
        print(f"Forskel i gennemsnitlige antal træk: {moves_diff:.2f} (95% CI: {moves_ci[0]:.2f}, {moves_ci[1]:.2f})")
        print(f"Forskel i gennemsnitlig AI-tid pr. træk: {ai_time_diff:.5f} "
              f"(95% CI: {ai_time_ci[0]:.5f}, {ai_time_ci[1]:.5f})")

        print("\nT-tests:")
        print(f"T-test for Move Counts: p-værdi = {ttest_moves.pvalue:.5f}")
        print(f"T-test for AI Move Times: p-værdi = {ttest_ai_times.pvalue:.5f}")

def save_game(results: dict, path: str) -> None:
    """Gem spilresultaterne til en tekstfil."""
    x = datetime.datetime.now()
    with open(path, "a") as f:
        f.write(f"\nResults for {x}:")
        f.write(f"\nWhite wins: {results['White']}")
        f.write(f"\nBlack wins: {results['Black']}")
        f.write(f"\nDraws: {results['Draw']}")
        f.write(f"\nMove Counts: {results.get('MoveCounts', [])}")
        f.write(f"\nAI Move Times: {results.get('AIMoveTimes', [])}")
        f.write("\n===================================\n")

def play_game(model: ChessCNN, board: chess.Board, stockfish: Stockfish) -> (str, int, list):
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

    # Board object.
    board = chess.Board()

    # Keeps track of the results.
    results = {"White": 0, "Black": 0, "Draw": 0, "MoveCounts": [], "AIMoveTimes": []}
    num_games = 10

    # Loop for playing games and saving results
    for i in range(num_games):
        print(f"Game: {i + 1}")
        result, length, ai_move_times = play_game(model, board, stockfish)
        results["MoveCounts"].append(length)
        results["AIMoveTimes"].append(ai_move_times)  # Store AI move times

        if result == "1-0":
            results["White"] += 1
        elif result == "0-1":
            results["Black"] += 1
        elif result == "1/2-1/2":
            results["Draw"] += 1

        board.reset()

    # Save game results to file
    save_game(results, "save_file.txt")

    # Analyze and compare saved results
    if compare:
        compare_versions_with_ttests("save_file.txt")