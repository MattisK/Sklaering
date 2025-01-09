from functions import parse_pgn
import numpy as np
import time


"""Generates the moves from PGN file games and saves it to a separate file."""
# TODO: memory optimise this also
start_time = time.time()
pgn_file_path = "lichess_db_standard_rated_2014-09.pgn"
moves = parse_pgn(pgn_file_path, 50000)
np.save("moves.npy", moves, allow_pickle=True)
print(f"Done making moves. Took {time.time() - start_time} seconds.")
