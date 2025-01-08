import torch
from ChessModel import ChessModel
import chess
import chess.engine
from functions import get_model_move
from stockfish import Stockfish


# Load the model.
model = ChessModel()
model.load_state_dict(torch.load("chess_model.pth"))

# Put model in evaluation mode.
model.eval()

stockfish_path = "C:/Users/chris/Desktop/Stockfish/stockfish/stockfish-windows-x86-64-avx2"
stockfish = Stockfish(stockfish_path)
stockfish.set_skill_level(20)

# Initilize a chess board.
board = chess.Board()

iteration = 0
while not board.is_game_over():
    iteration += 1

    # Check if it's white's turn.
    if board.turn == chess.WHITE:
        model_move = get_model_move(board, model)
        if model_move:
            print(model_move)
            board.push(model_move)
            print(board)
            stockfish.set_fen_position(board.fen())
    else:
        stockfish_best_move = stockfish.get_best_move()
        stockfish_move = chess.Move.from_uci(stockfish_best_move)
        if stockfish_move:
            print(stockfish_move)
            board.push(stockfish_move)
            print(board)

print("Checkmate:", board.is_checkmate())
print("Game over:", board.is_game_over())

result = board.result()
if result == "1-0":
    print("White wins")
elif result == "0-1":
    print("Black wins")
elif result == "1/2-1/2":
    print("It's a draw")

print(iteration)
