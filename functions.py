import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import chess
import random
from ReplayMemory import ReplayMemory
from DQN import DQN


steps_done = 0


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Takes the board state and converts it to a one-hot tensor.
    """
    # An 8x8 board with a third dimension of size 12 to separate the pieces into different planes.
    np_board = np.zeros((12, 8, 8))

    # Gives a value to each of the types of pieces.
    piece_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
                 "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11}
    
    # Inserts a 1 in each of the pieces' planes where they are located.
    for square, piece in board.piece_map().items():
        plane = piece_map[str(piece)]
        row, col = divmod(square, 8)
        np_board[plane, row, col] = 1
    
    # Returns a tensor of the one-hot
    return torch.tensor(np_board, dtype=torch.float32)


def choose_action(state: torch.Tensor, policy_net: DQN, n_actions: int, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=200):
    global steps_done
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
    steps_done += 1

    if random.random() < epsilon:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)  # Random action
    else:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)  # Greedy action


def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma=0.99):
    if len(memory) < batch_size:
        return

    # Sample a batch of transitions
    transitions = memory.sample(batch_size)
    batch = memory.Transition(*zip(*transitions))

    # Separate non-terminal states
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    
    # Stack states, actions, and rewards
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s') for non-terminal states
    next_state_values = torch.zeros(batch_size)
    if non_final_next_states.size(0) > 0:  # Avoid empty tensor errors
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    board = chess.Board()

    print(board_to_tensor(board))