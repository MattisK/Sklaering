import chess
import chess.pgn
import time


# Initialize the board.
board = chess.Board()

# Path to Lichess dataset.
file_path = "lichess_db_standard_rated_2014-09.pgn"

# Counter that increases by one after each game.
counter = 0

# Used to chech if the game should be slowed down for easier debugging.
slow = True

# Number of total games to run.
games = 10

# Open the dataset.
with open(file_path, "r", encoding="utf-8") as pgn_file:
    # While loop that runs as long as the game number is less than or equal to the total number of games.
    while counter <= games:
        # Read the complete game in chess notation and make sure it is not None.
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        
        # Looks at each of the move made in the complete chess game and visualizes the play.
        for move in game.mainline_moves():
            board.push(move)
            print(board)
            print("---------------")
            
            # If slow is set to True, slows down the game.
            if slow:
                time.sleep(1)
        
        # Info on game termination.
        print(game.headers.get("Termination"))
        print(game.headers.get("Result"))

        # Reset the board and increase counter by one.
        board.reset()
        counter += 1
