import chess.pgn
import chess.svg
from sympy.benchmarks.bench_discrete_log import data_set_1
from torch.utils.data import Dataset
from chessboard import *

class ChessDataset():
    def __init__(self):
        """
        Initializes the dataset by reading a PGN file and extracting positions and results.
        Each position is encoded as an 8x8x12 list and each result is a single number.
        Each index in the positions list corresponds to the same index in the results list.
        results can have the values 1 (white wins), -1 (black wins) or 0 (draw).
        """
        self.positions = [] # List of positions in one-hot encoded format
        self.results = [] # List of game outcomes (1 for white win, -1 for black win, 0 for draw) corresponding to the positions

        file_path = "lichess_db_standard_rated_2014-09.pgn" #relative path
        counter = 0

        with open(file_path, "r", encoding="utf-8") as pgn_file: #open file
            while counter < 1000: #read 1000 games
                game = chess.pgn.read_game(pgn_file)
                if game is None: #if no more games,
                    break

                # Get game result
                result = game.headers.get("Result")
                if result == "1-0":
                    outcome = 1  # white wins
                elif result == "0-1":
                    outcome = -1  # black wins
                elif result == "1/2-1/2":
                    outcome = 0  # Remis, they are both bad
                else:
                    continue #skip the game if it is not one of the above however this should not happen

                # Extract positions from the game
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    one_hot_position = board_to_one_hot(board)
                    self.positions.append(one_hot_position)
                    self.results.append(outcome)

                counter += 1
        print("Dataset initialized with {} positions".format(len(self.positions)))

    def __getitem__ (self, index):
        """
        Returns the position and result at the given index.
        Allows the dataset to be accessed with the index operator like so: dataset[42] = (position, result)
        """
        return self.positions[index], self.results[index]



dataset = ChessDataset()

