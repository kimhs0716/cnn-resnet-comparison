import os
import sys
from pathlib import Path

if os.getcwd().endswith("scripts"):
    os.chdir("..")
sys.path.insert(0, "src")

import yaml
import torch

from data import get_loaders
from models import PlainCNN
from trainer import train, evaluate
from utils import resolve_device, set_seed, plot_history


def main(cfg_path="configs.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["device"])
    print(f"Device: {device}")

    seed = cfg.get("seed", 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    train_loader, val_loader, test_loader = get_loaders(
        data_dir="data",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["train"]["augment"],
        val_split_seed=cfg["train"]["val_split_seed"],
    )
    print(f"Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)} batches")

    model = PlainCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    epochs = cfg["train"]["epochs"]
    ckpt_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints"))

    history = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        ckpt_dir=ckpt_dir,
        save_name="best_plain_cnn.pt",
    )

    model.load_state_dict(torch.load(ckpt_dir / "best_plain_cnn.pt"))
    test_loss, test_acc = evaluate(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    plot_history(history, save_path=ckpt_dir / "history_plain_cnn.png", show=False)


if __name__ == "__main__":
    main("configs.yaml")
