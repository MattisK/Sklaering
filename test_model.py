import chess
from ChessCNN import ChessCNN
from functions import get_best_move
from stockfish import Stockfish
import torch
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    # Load the trained model.
    model = ChessCNN()
    model.load_state_dict(torch.load("chess_model.pth"))

    # Set the model to evaluation mode.
    model.eval()

    # Initialize stockfish and set the skill level.
    stockfish_path = "C:/Users/chris/Desktop/Stockfish/stockfish/stockfish-windows-x86-64-avx2"
    stockfish = Stockfish(stockfish_path, depth=1)
    stockfish.set_skill_level(0)

    # Board object.
    board = chess.Board()
    
    # Keeps track of the results.
    results = {"White": 0, "Black": 0, "Draw": 0}
    num_games = 100
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
    return results

def plot_results(results):
    # Extract data for plotting
    parameter_a = np.array([results["White"], results["Black"], results["Draw"]])
    parameter_b = np.array([0, 1, 2])  # Dummy parameter for plotting

    # Plot parameters for each iteration on contour plot of cost
    plt.plot(parameter_b, parameter_a, '.-')
    a_mesh = np.linspace(-.2, 1, 400)
    b_mesh = np.linspace(-.2, 1.75, 400)
    B, A = np.meshgrid(b_mesh, a_mesh)
    Z = np.sum((A[:,:,np.newaxis]*parameter_b[np.newaxis,np.newaxis]+B[:,:,np.newaxis]-parameter_a[np.newaxis,np.newaxis])**2,2)
    plt.contour(B, A, np.log(Z), 7)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('b')
    plt.ylabel('a')
    plt.grid(True)
    plt.title('Cost contour plot')
    plt.show()

if __name__ == "__main__":
    results = main()
    plot_results(results)