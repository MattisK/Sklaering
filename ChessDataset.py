import chess.pgn
from torch.utils.data import Dataset
import chess
import torch
from functions import encode_board, encode_move


class ChessDataset(Dataset):
    def __init__(self, pgn_path: str, max_games: int) -> None:
        """
        The ChessDataset class is designed to handle and preprocess chess game data for use in machine learning models.
        It inherits from the abstract class Dataset, which is typically used in the PyTorch framework for handling datasets.
        Takes a path to a PGN (Portable Game Notation) file as input.
        Initializes the pgn_path attribute with the provided path.
        Calls the load_games method to load chess games from the PGN file and stores them in the games attribute.
        """
        self.pgn_path = pgn_path
        self.max_games = max_games
        self.game_offsets = self.index_pgn_file()

    def index_pgn_file(self) -> list[chess.pgn.Game]:
        """Index a number of games with specific traits from a PGN file. Returns a list of all indexed games."""
        offsets = []
        counter = 0
        
        # Open PGN file.
        with open(self.pgn_path, "r") as pgn_file:
            # While loop runs through a number of games in the PGN file.
            while True:
                # Breaks the loop if the amount of games exceeds the max amount of games.
                if self.max_games and len(offsets) >= self.max_games:
                    break

                # Load the game
                offset = pgn_file.tell()
                game = chess.pgn.read_game(pgn_file)

                # Break the loop if the game is None.
                if game is None:
                    break
                
                # Append to the list of all offsets, if the following traits are fullfiled.
                try:
                    if game.headers.get("Termination") == "Normal" and game.headers.get("Result") == "1-0" and (int(game.headers.get("WhiteElo")) >= 2400 and int(game.headers.get("BlackElo")) >= 2400):
                        offsets.append(offset)
                        counter += 1

                        if counter % 10 == 0:
                            print(f"Indexed {counter} games.")
                except ValueError:
                    continue

        print(f"Done indexing. Games indexed: {len(offsets)}")
        
        return offsets
    

    def __len__(self) -> int:
        """
        Required for the abstract class 'Dataset'. Returns the amount of games indexed.
        """
        return len(self.game_offsets)
    

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Required for the abstract class 'Dataset'.
        Loads and returns an encoded sample from the dataset at the given index 'idx'. Returns a tuple of tensors.
        """
        # Load a game at a given index.
        with open(self.pgn_path, "r") as pgn_file:
            pgn_file.seek(self.game_offsets[idx])
            game = chess.pgn.read_game(pgn_file)

        # Load a game and the board state for that game.
        board = game.board()

        # Empty lists for later use.
        positions = []
        moves = []

        # Looks at each move in the game and saves the board state and move,
        # then updates the board by pushing the move to the board.
        for move in game.mainline_moves():
            positions.append(encode_board(board.copy()))
            moves.append(encode_move(move))
            board.push(move)

        # If we have a situation where there are no moves in the game we look at the next index.
        if not positions or not moves:
            return self.__getitem__((idx + 1) % len(self.game_offsets))

        # Returns a tuple of tensors for the boards and moves.
        return torch.stack(positions), torch.tensor(moves, dtype=torch.long)
