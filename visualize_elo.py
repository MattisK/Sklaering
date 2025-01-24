import matplotlib.pyplot as plt
import numpy as np


def visualize_elo(wins: np.ndarray, draws: np.ndarray) -> None:
    x_axis = np.arange(len(wins))
    plt.bar(x_axis - 0.2, wins, 0.4, label="Wins")
    plt.bar(x_axis + 0.2, draws, 0.4,  label="Draws")
    plt.xlabel("Elo")
    plt.ylabel("Amount")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    wins = np.load("All elo data/wins.npy", allow_pickle=True)
    draws = np.load("All elo data/draws.npy", allow_pickle=True)

    visualize_elo(wins, draws)
