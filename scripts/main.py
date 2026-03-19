import json
import os
import sys
from pathlib import Path
import time

if os.getcwd().endswith("scripts"):
    os.chdir("..")
sys.path.insert(0, "src")

import yaml
import torch

from data import get_loaders
from models import PlainCNN, ResNet, PlainCNNDeep, ResNetDeep
from trainer import train, evaluate
from utils import resolve_device, set_seed, plot_history


EXPERIMENTS = {
    "E1": {"model_cls": PlainCNN,     "augment": False, "save_name": "best_plain_cnn.pt"},
    "E2": {"model_cls": ResNet,       "augment": False, "save_name": "best_resnet.pt"},
    "E3": {"model_cls": PlainCNN,     "augment": True,  "save_name": "best_plain_cnn_aug.pt"},
    "E4": {"model_cls": ResNet,       "augment": True,  "save_name": "best_resnet_aug.pt"},
    "E5": {"model_cls": PlainCNNDeep, "augment": True,  "save_name": "best_plain_cnn_deep.pt"},
    "E6": {"model_cls": ResNetDeep,   "augment": True,  "save_name": "best_resnet_deep.pt"},
}


def run_experiment(exp_id, cfg, device):
    exp = EXPERIMENTS[exp_id]
    print(f"\n{'='*50}")
    print(f"Experiment {exp_id}: {exp['model_cls'].__name__} | augment={exp['augment']}")
    print(f"{'='*50}")

    train_loader, val_loader, test_loader = get_loaders(
        data_dir="data",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=exp["augment"],
        val_split_seed=cfg["train"]["val_split_seed"],
    )
    print(f"Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)} batches")

    model = exp["model_cls"](num_classes=cfg["model"]["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints"))
    result_dir = Path(cfg["train"].get("result_dir", "results"))
    ckpt_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    epochs = cfg["train"]["epochs"]

    history = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        ckpt_dir=ckpt_dir,
        save_name=exp["save_name"],
    )

    best_val_acc = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best_val_acc) + 1
    print(f"Loading best model for testing from Epoch {best_epoch} with Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(ckpt_dir / exp["save_name"]))
    test_loss, test_acc = evaluate(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    plot_history(history, save_path=result_dir / exp["save_name"].replace(".pt", ".png"), show=False)

    return test_loss, test_acc, best_epoch


def main(cfg_path="configs.yaml", experiments=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["device"])
    print(f"Device: {device}")

    seed = cfg.get("seed", 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    if experiments is None:
        experiments = list(EXPERIMENTS.keys())

    metrics = {}

    for exp_id in experiments:
        start = time.time()
        test_loss, test_acc, best_epoch = run_experiment(exp_id, cfg, device)
        metrics[exp_id] = {
            "test_loss": test_loss, 
            "test_acc": test_acc, 
            "best_epoch": best_epoch,
            "duration": time.time() - start
        }

    print(f"\n{'='*50}")
    print("Final Results:")
    print(f"{'='*50}")
    for exp_id, metric in metrics.items():
        print(
            f"{exp_id}: Test Loss: {metric['test_loss']:.4f} | "
            f"Test Acc: {metric['test_acc']:.4f} | "
            f"Best Epoch: {metric['best_epoch']} | "
            f"Duration: {metric['duration']:.2f}s"
        )

    result_dir = Path(cfg["train"].get("result_dir", "results"))
    save_path = result_dir / "final_metrics.json"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main("configs.yaml")
