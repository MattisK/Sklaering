import chess


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    
    def reset(self):
        self.board.reset()
        