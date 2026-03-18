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


def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()