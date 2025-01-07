import chess
import torch
import numpy as np


class ChessEnv:
    def __init__(self): # on init create a new board.
        self.board = chess.Board()

    
    def reset(self): # reset the board and return the state in fen notation.
        self.board.reset()
        return self.get_state()
    

    def actions(self): # The complete list of legal moves.
        return list(self.board.legal_moves)


    def get_state(self): # Returns the state of the board as a torch tensor. 
        piece_map = self.board.piece_map()
        state = np.zeros(64, dtype=int)
        for square, piece in piece_map.items():
            if str(piece.symbol()) == str(piece.symbol()).lower():
                state[square] = -1 * piece.piece_type
            else:
                state[square] = piece.piece_type
        return torch.tensor(state, dtype=torch.float32)


if __name__ == "__main__":
    env = ChessEnv()
    print(env.get_state())
        