import chess


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    
    def reset(self):
        self.board.reset()
        return self.get_state()
    

    def get_state(self):
        pass