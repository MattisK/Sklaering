from functions import parse_pgn
import numpy as np


"""Generates the moves from PGN file games and saves it to a separate file."""


pgn_file_path = "lichess_db_standard_rated_2014-09.pgn"
moves = parse_pgn(pgn_file_path, 100)
np.save("moves.npy", moves, allow_pickle=True)
