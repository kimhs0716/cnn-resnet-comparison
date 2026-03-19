import random

import torch

import numpy as np
import matplotlib.pyplot as plt


def resolve_device(device):
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            print("[WARNING] CUDA is not available, falling back to XPU")
            return "xpu"
        print("[WARNING] CUDA is not available, falling back to CPU")
        return "cpu"
    if device == "xpu":
        if torch.xpu.is_available():
            return "xpu"
        print("[WARNING] XPU is not available, falling back to CPU")
        return "cpu"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Invalid device: {device}")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def plot_history(history, save_path=None, show=True):
    plt.figure(figsize=(12, 5))

    best_val_loss = min(history["val_loss"])
    best_epoch = history["val_loss"].index(best_val_loss) + 1
    best_val_acc = history["val_acc"][best_epoch - 1]

    epoch_range = range(1, len(history["train_loss"]) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, history["train_loss"], label="Train Loss")
    plt.plot(epoch_range, history["val_loss"], label="Validation Loss")
    plt.axhline(y=best_val_loss, color="r", linestyle="--", label=f"Best Val Loss: {best_val_loss:.4f}")
    plt.axvline(x=best_epoch, color="r", linestyle="--", label=f"Best Epoch: {best_epoch}")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, history["train_acc"], label="Train Accuracy")
    plt.plot(epoch_range, history["val_acc"], label="Validation Accuracy")
    plt.axhline(y=best_val_acc, color="r", linestyle="--", label=f"Val Acc at Best Epoch: {best_val_acc:.4f}")
    plt.axvline(x=best_epoch, color="r", linestyle="--", label=f"Best Epoch: {best_epoch}")
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()