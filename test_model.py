import torch
from ChessModel import ChessModel
import chess
from functions import board_to_tensor, move_to_idx


# Load the model.
model = ChessModel()
model.load_state_dict(torch.load("chess_model.pth"))

# Initilize a chess board.
board = chess.Board()

iteration = 0
while True:
    iteration += 1

    # Board tensor
    input_tensor = board_to_tensor(board).unsqueeze(0)

    # Put model in evaluation mode.
    model.eval()

    # Disables gradient computation for memory efficiency and shorter duration.
    with torch.no_grad():
        # Feed the board tensor to the model (predicted moves).
        outputs = model(input_tensor)

        # Finds the index with the highest value in outputs.
        predicted_idx = torch.argmax(outputs, dim=1).item()

    # Convert the predicted index back into a move.
    move_idx = move_to_idx(board)
    idx_to_move = {idx: move for move, idx in move_idx.items()}
    predicted_move_uci = idx_to_move[predicted_idx]
    predicted_move = chess.Move.from_uci(predicted_move_uci)

    # Print predicted move if it is legal and make the play on the board. If it is not legal, brak the loop.
    if predicted_move in board.legal_moves:
        print(predicted_move)
        board.push(predicted_move)
    else:
        print("Ended at:", iteration)
        break