import chess.pgn
from torch.utils.data import Dataset
import chess
import numpy as np
import torch
from functions import encode_board, encode_move


class ChessDataset(Dataset):
    # The ChessDataset class is designed to handle and preprocess chess game data for use in machine learning models.
    # It inherits from the abstract class Dataset, which is typically used in the PyTorch framework for handling datasets.
    # Takes a path to a PGN (Portable Game Notation) file as input.
    # Initializes the pgn_path attribute with the provided path.
    # Calls the load_games method to load chess games from the PGN file and stores them in the games attribute.

    def __init__(self, pgn_path: str) -> None:
        """
        A class that inherits from the abstract class 'Dataset',
        which makes the model store the samples and their corresponding labels.
        Initialize the PGN-path and the list of games.
        """
        self.pgn_path = pgn_path
        self.games = self.load_games()


    # Opens the PGN file and reads up to 100 chess games.
    # Stores each game in a list and returns this list.
    def load_games(self) -> list[chess.pgn.Game]:
        """Load all games with specific traits from a PGN file. Returns a list of all games loaded."""
        games = []
        counter = 0
        
        # Open PGN file.
        with open(self.pgn_path, "r") as pgn_file:
            # While loop runs through all the games in the PGN file.
            while counter < 1000:
                game = chess.pgn.read_game(pgn_file)

                # Break the loop if the game is None.
                if game is None:
                    break
                
                # Append to the list of all games.
                if game.headers.get("Termination") == "Normal" and game.headers.get("Result") == "1-0":
                    games.append(game)
                    counter += 1
                
                    if counter % 1000 == 0:
                        print(f"Loaded {counter} games.")

        print(f"Done loading. Loaded {counter} games in total.")
        
        return games
    

    def __len__(self) -> int:
        """
        Required for the abstract class 'Dataset'. Returns the amount of games loaded.
        """
        return len(self.games)
    

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Required for the abstract class 'Dataset'.
        Loads and returns a sample from the dataset at the given index 'idx'. Returns a tuple of tensors.
        """
        # Takes an index idx and retrieves the corresponding game from the games list.
        # Extracts the board state and moves for the game.
        # If there are no moves, it recursively calls itself with the next index.
        # Determines the result of the game (1.0 for white win, -1.0 for black win, 0.0 for draw) and assigns this result to all moves.
        # Randomly selects a move and its corresponding board state and result.
        # Encodes the board state and move into tensors.
        # Returns a tuple of tensors representing the board state, move, and result.

        # Load a game and the board state for that game.
        game = self.games[idx]
        board = game.board()

        # Empty lists for later use.
        positions = []
        moves = []

        # Looks at each move in the game and saves the board state and move,
        # then updates the board by pushing the move to the board.
        for move in game.mainline_moves():
            positions.append(board.copy())
            moves.append(move)
            board.push(move)

        # If we have a situation where there are no moves in the game we look at the next index.
        if not positions or not moves:
            return self.__getitem__((idx + 1) % len(self.games))

        # Randomly selects an index of max size of the positions. Uses this to fetch a board, move, and result
        # and ensures better sampling of the training data.
        random_idx = np.random.randint(len(positions))
        board = positions[random_idx]
        move = moves[random_idx]

        # Encodes the board as a 12x8x8 tensor and the move as an integer.
        board_encoded = encode_board(board)
        move_encoded = encode_move(move)

        # Returns a tuple of tensors for the board, move and result.
        return board_encoded, torch.tensor(move_encoded, dtype=torch.float32)
