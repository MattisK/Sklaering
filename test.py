from functions import *
board1 = chess.Board("8/8/8/4p1K1/2k1P3/8/8/8 b - - 0 1")
move = chess.Move.from_uci("c4d4")



print("material", reward(board1, move, "material"))
print("check", reward(board1, move, "check"))
print("stockfish", reward(board1, move, "stockfish"))
print("human", reward(board1, move, "human"))