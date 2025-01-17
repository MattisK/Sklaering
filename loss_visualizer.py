import matplotlib.pyplot as plt
import numpy as np


def visualize_loss(train_loss_history: list[float], val_loss_history: list[float]) -> None:
    """Visualize the training and validation loss."""
    plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, label="Training loss")
    plt.plot(np.arange(1, len(val_loss_history) + 1), val_loss_history, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_loss_history = list(np.load("loss_history.npy", allow_pickle=True))
    val_loss_history = list(np.load("val_loss_history.npy", allow_pickle=True))

    visualize_loss(train_loss_history, val_loss_history)
    