import chess
import chess.pgn
import chess.svg

board = chess.Board() 
        
def reset_board(): # resets board to starting position
    board = chess.Board()
    return board
    
def chess_to_matrix(move):

    row = ord(move[0]) - ord('a') 
    column = 8 - int(move[1])
    return row, column
    
def apply_move(move, board):
    """
    Applies a list of SAN moves to an 8x8 matrix representation of the board.
    """
    board.push_san(move)  # Apply the move to the python-chess board

    # Optionally, convert to your 8x8 matrix
    matrix_board = convert_chess_board_to_matrix(board)
    
    return matrix_board

def convert_chess_board_to_matrix(board):
    """
    Converts a python-chess board object to an 8x8 matrix representation.
    """
    matrix = [[0 for _ in range(8)] for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)  # Convert from 0-63 to row index
            col = square % 8        # Convert from 0-63 to column index
            matrix[row][col] = piece.symbol()  # Use piece.symbol() for notation
    return matrix


file_path = "lichess_db_standard_rated_2014-09.pgn"
counter = 0

with open(file_path, "r", encoding="utf-8") as pgn_file:
    while counter == 0:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        #
        #apply_san_moves("Nf3", board)
        apply_move("e4", board)
        print(board)
        #print(game.headers.get("Result"))
        #print(game.mainline_moves())
        #print(counter)
        #print(game.headers.get("Termination"))

        counter += 1
        

