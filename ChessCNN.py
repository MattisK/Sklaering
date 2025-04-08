import torch.nn as nn
import torch
import chess


class ChessCNN(nn.Module):
    def __init__(self) -> None:
        """
        Convolutional Neural Network (CNN) for the chess model.
        Initialize the layers of the chess model.
        """
        super(ChessCNN, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(13, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Convolutional layer 2, will be called multiple times.
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Policy network. Convolutional and linear layers.
        self.net = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64 * 64 + 64),
        )
    
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        From the documentation: Defines the computation performed at every call.
        """
        x = self.conv1(x)
        
        for _ in range(6):
            x = self.conv2(x)
        
        policy = self.net(x)

        return policy
    
    def play(self, board, limit):
        # Convert the board to a tensor
        board_tensor = self.board_to_tensor(board)
        board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension

        # Get the policy from the model
        with torch.no_grad():
            policy = self.forward(board_tensor)

        # Convert policy to move probabilities
        move_probs = policy.squeeze().softmax(dim=0)

        # Get legal moves
        legal_moves = list(board.legal_moves)

        # Map legal moves to their probabilities
        move_prob_dict = {}
        for move in legal_moves:
            move_index = self.move_to_index(move)
            move_prob_dict[move] = move_probs[move_index].item()

        # Select the move with the highest probability
        best_move = max(move_prob_dict, key=move_prob_dict.get)

        return chess.engine.PlayResult(best_move)

    def board_to_tensor(self, board):
        # Convert the chess board to a tensor representation
        # This is a placeholder implementation
        board_tensor = torch.zeros((13, 8, 8))
        # Fill in the tensor with the board state
        return board_tensor

    def move_to_index(self, move):
        # Convert a chess move to an index in the policy output
        # This is a placeholder implementation
        move_index = 0
        # Calculate the index based on the move
        return move_index
    
    def predict(self, board):
        # Convert the board to a tensor
        board_tensor = self.board_to_tensor(board)
        board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension

        # Get the policy from the model
        with torch.no_grad():
            policy = self.forward(board_tensor)

        # Convert policy to move probabilities
        move_probs = policy.squeeze().softmax(dim=0)

        # Get legal moves
        legal_moves = list(board.legal_moves)

        # Map legal moves to their probabilities
        move_prob_dict = {}
        for move in legal_moves:
            move_index = self.move_to_index(move)
            move_prob_dict[move] = move_probs[move_index].item()

        # Select the move with the highest probability
        best_move = max(move_prob_dict, key=move_prob_dict.get)

        return best_move
