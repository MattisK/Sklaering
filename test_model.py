import torch
from ChessModel import ChessModel
import chess
import chess.engine
from functions import get_model_move
from stockfish import Stockfish


# Counter for the total amount of random moves.
random_counter_total = 0

# Load the model.
model = ChessModel()
model.load_state_dict(torch.load("chess_model.pth"))

# Put model in evaluation mode.
model.eval()

stockfish_path = "C:/Users/chris/Desktop/Stockfish/stockfish/stockfish-windows-x86-64-avx2"
stockfish = Stockfish(stockfish_path, depth=1)
stockfish.set_skill_level(0)

# Initilize a chess board.
board = chess.Board()


def play_game(board: chess.Board) -> int:
    global random_counter_total
    while not board.is_game_over():

        # Check if it's white's turn.
        if board.turn == chess.WHITE:
            model_move, random_counter = get_model_move(board, model)
            if model_move:
                board.push(model_move)
                stockfish.set_fen_position(board.fen())
                random_counter_total += random_counter
        else:
            stockfish_best_move = stockfish.get_best_move()
            stockfish_move = chess.Move.from_uci(stockfish_best_move)
            if stockfish_move:
                board.push(stockfish_move)


results = {"White": 0, "Black": 0, "Draw": 0}
num_games = 10
for i in range(num_games):
    print("Game:", i)
    play_game(board)
    result = board.result()
    if result == "1-0":
        results["White"] += 1
    elif result == "0-1":
        results["Black"] += 1
    elif result == "1/2-1/2":
        results["Draw"] += 1

    board.reset()

print(results)
print(random_counter_total / num_games)