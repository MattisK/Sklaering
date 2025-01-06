

def extract_moves(game):
    """
    Function to extract the moves from the complete list of moves in chess notation. Returns a list of tuples of the turn number, white moves and black moves.
    """
    # Converts the chess game to a string and splits it by a space.
    moves_str = str(game.mainline_moves())
    split_str = moves_str.split(" ")

    # Takes the every 1st, 4th, 7th... element and converts it to an int (and removes the dot). This is the turn number.
    turns = split_str[::3]
    turns_clean = [int(element.rstrip(".")) for element in turns]

    # Takes every 2nd, 5th, 8th... element as the white moves.
    white_moves = split_str[1::3]

    # Takes every 3rd, 6th, 9th... element as the black moves.
    black_moves = split_str[2::3]

    # Zips the values and converts it to a list of tuples. Returns this list
    return list(zip(turns_clean, white_moves, black_moves))


"""# Extracts the moves from the game in a tuple.
        extracted_moves = extract_moves(game)

        # For each turn, checks if the element is the turn number, if a move was made, and then makes the move and prints the board.
        for turn in extracted_moves:
            for move in turn:
                if type(move) == int:
                    continue
                elif move:
                    board.push_san(move)
                    print(board)
                    print("---------------")
                    
                    # If slow is set to true, slows down the game.
                    if slow:
                        time.sleep(1)"""
