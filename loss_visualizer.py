import matplotlib.pyplot as plt
import numpy as np


def visualize_loss(train_loss_history: np.ndarray, val_loss_history: np.ndarray) -> None:
    """Visualize the training and validation loss."""
    plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, label="Training loss")
    plt.plot(np.arange(1, len(val_loss_history) + 1), val_loss_history, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    train_loss_history = np.load("loss_history.npy", allow_pickle=True)
    val_loss_history = np.load("val_loss_history.npy", allow_pickle=True)

    visualize_loss(train_loss_history, val_loss_history)
    