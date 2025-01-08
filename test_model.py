import torch
from ChessModel import ChessModel
import chess
from functions import board_to_tensor, move_to_idx
import random


# Load the model.
model = ChessModel()
model.load_state_dict(torch.load("chess_model.pth"))

# Initilize a chess board.
board = chess.Board()

iteration = 0
while not board.is_game_over():
    iteration += 1

    # Board tensor.
    input_tensor = board_to_tensor(board).unsqueeze(0)

    # Put model in evaluation mode.
    model.eval()

    # Disables gradient computation for memory efficiency and shorter duration.
    with torch.no_grad():
        # Feed the board tensor to the model (predicted moves).
        outputs = model(input_tensor)

        top_k = torch.topk(outputs, k=5, dim=1)
        predicted_indices = top_k.indices[0].tolist()
        print(predicted_indices)

    move_idx = move_to_idx(board)
    idx_to_move = {idx: move for move, idx in move_idx.items()}

    print(idx_to_move)

    bool_list = []

    for predicted_idx in predicted_indices:
        if predicted_idx in idx_to_move:
            predicted_move_uci = idx_to_move[predicted_idx]
            predicted_move = chess.Move.from_uci(predicted_move_uci)

            # Print predicted move if it is legal and make the play on the board.
            if predicted_move in board.legal_moves:
                print(predicted_move)
                board.push(predicted_move)
                print(board)
                bool_list.append(True)
                break
    else:
        if not any(bool_list):
            random_move = random.choice(list(idx_to_move.keys()))
            random_move_uci = idx_to_move[random_move]
            random_move_final = chess.Move.from_uci(random_move_uci)
            board.push(random_move_final)
            print(board)

    print(board.is_game_over())
    print(board.is_checkmate())
    print(board.is_fifty_moves())
    print(board.is_variant_draw())
    print(board.can_claim_draw())
