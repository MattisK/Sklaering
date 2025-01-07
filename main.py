import random
import math
import torch
import torch.optim as optim
from DQN import DQN
from ChessEnv import ChessEnv
from ReplayMemory import ReplayMemory
import chess
import chess.pgn

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

env = ChessEnv()

n_actions = 64 * 64
state = env.reset() 
n_observations = 64

EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
LR = 1e-4

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state): # selects the action to take based on the state of the board
    global steps_done # sets the varible
    sample = random.random() # random number between 0 and 1
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY) # epsilon greedy strategy
    steps_done += 1 # update 1

    q_values = policy_net(state)
    legal_moves_idxs = [move.from_square * 64 + move.to_square for move in env.actions()] # get the legal moves
    best_move_idx = legal_moves_idxs[torch.argmax(q_values[0][legal_moves_idxs])] # get the best move
    return chess.Move.from_uci(f"{best_move_idx // 64}{best_move_idx % 64}")
    """else:
        return random.choice(env.actions())"""


print(select_action(state))

"""
def select_action(state):
    global steps_done
    sample = random.random()
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1

    q_values = policy_net(state)

    # Generér alle mulige træk på et skakbræt (64 felter x 64 felter)
    all_moves_idxs = [from_square * 64 + to_square for from_square in range(64) for to_square in range(64)]
    
    if sample > epsilon_threshold:
        # Vælg det bedste træk baseret på Q-værdier
        best_move_idx = all_moves_idxs[torch.argmax(q_values[0][all_moves_idxs])]
    else:
        # Vælg et tilfældigt træk
        best_move_idx = random.choice(all_moves_idxs)

    # Konverter trækket til UCI-format (Universal Chess Interface)
    from_square = best_move_idx // 64
    to_square = best_move_idx % 64
    return chess.Move(from_square, to_square)
"""