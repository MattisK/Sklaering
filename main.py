import random
import math
import torch
import torch.optim as optim
from ChessEnv import ChessEnv
from DQN import DQN
from functions import board_to_tensor, optimize_model, choose_action
from ReplayMemory import ReplayMemory
import chess


BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-4
NUM_EPISODES = 1000

n_actions = 4096
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

for episode in range(NUM_EPISODES):
    board = chess.Board()
    state = board_to_tensor(board).unsqueeze(0)  # Add batch dimension

    for t in range(200):  # Limit moves per game
        action = choose_action(state, policy_net, n_actions)
        move_index = action.item()
        from_square = move_index // 64
        to_square = move_index % 64
        uci_move = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"

        try:
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                raise ValueError("Illegal move")
            board.push(move)
            reward = 1.0 if board.is_checkmate() else 0.0
            next_state = board_to_tensor(board).unsqueeze(0)  # Add batch dimension
            done = board.is_game_over()
        except:
            reward = -1.0
            next_state = None
            done = True

        reward = torch.tensor([reward], dtype=torch.float32)
        memory.push(state, action, next_state, reward)

        if done:
            next_state = None  # No next state if the game is over
        else:
            next_state = board_to_tensor(board).unsqueeze(0)  # Add batch dimension

        state = next_state

        optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA)

        if done:
            break

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode + 1}/{NUM_EPISODES} complete")

print("Training complete")
