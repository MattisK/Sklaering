import chess
import torch
import numpy as np
from functions import board_to_tensor


class ChessEnv:
    def __init__(self): # on init create a new board.
        self.board = chess.Board()

    
    def reset(self): # reset the board and return the state in fen notation.
        self.board.reset()
        return self.get_state()
    

    def actions(self): # The complete list of legal moves.
        return list(self.board.legal_moves)


    def get_state(self): # Returns the state of the board as a torch tensor. 
        return board_to_tensor(self.board)


if __name__ == "__main__":
    env = ChessEnv()
    print(env.get_state())
        