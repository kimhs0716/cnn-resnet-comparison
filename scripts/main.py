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
from models import PlainCNN, ResNet
from trainer import train, evaluate
from utils import resolve_device, set_seed, plot_history, save_history


EXPERIMENTS = {
    "E1": {"model_cls": PlainCNN, "num_blocks": 1, "augment": False, "save_name": "plain_cnn"},
    "E2": {"model_cls": ResNet,   "num_blocks": 1, "augment": False, "save_name": "resnet"},
    "E3": {"model_cls": PlainCNN, "num_blocks": 1, "augment": True,  "save_name": "plain_cnn_aug"},
    "E4": {"model_cls": ResNet,   "num_blocks": 1, "augment": True,  "save_name": "resnet_aug"},
    "E5": {"model_cls": PlainCNN, "num_blocks": 3, "augment": True,  "save_name": "plain_cnn_deep"},
    "E6": {"model_cls": ResNet,   "num_blocks": 3, "augment": True,  "save_name": "resnet_deep"},
}


def run_experiment(exp_id, cfg, device):
    exp = EXPERIMENTS[exp_id]
    print(f"\n{'='*50}")
    print(f"Experiment {exp_id}: {exp['model_cls'].__name__} | num_blocks={exp['num_blocks']} | augment={exp['augment']}")
    print(f"{'='*50}")

    dataset = cfg["data"]["dataset"]
    ckpt_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints")) / dataset
    result_dir = Path(cfg["train"].get("result_dir", "results")) / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_name = exp["save_name"]
    epochs = cfg["train"]["epochs"]

    train_loader, val_loader, test_loader = get_loaders(
        dataset=dataset,
        data_dir="data",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=exp["augment"],
        val_split_seed=cfg["train"]["val_split_seed"],
    )
    print(f"Selected Dataset: {dataset}")
    print(f"Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)} batches")

    model = exp["model_cls"](num_classes=cfg["model"]["num_classes"], num_blocks=exp["num_blocks"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    history = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        ckpt_dir=ckpt_dir,
        save_name=f"best_{save_name}.pt",
    )

    best_val_loss = min(history["val_loss"])
    best_epoch = history["val_loss"].index(best_val_loss) + 1
    print(f"Loading best model for testing from Epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(ckpt_dir / f"best_{save_name}.pt"))
    test_loss, test_acc = evaluate(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    plot_history(history, save_path=result_dir / f"{save_name}.png", show=False)
    save_history(history, save_path=result_dir / f"history_{save_name}.json")

    result = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "num_params": num_params,
    }

    return result


def main(cfg_path="configs.yaml", experiments=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["device"])
    print(f"Device: {device}")

    seed = cfg.get("seed", 42)
    print(f"Random seed: {seed}")

    if experiments is None:
        experiments = list(EXPERIMENTS.keys())

    metrics = {}

    for exp_id in experiments:
        set_seed(seed)
        start = time.time()
        result = run_experiment(exp_id, cfg, device)
        metrics[exp_id] = {
            "test_loss": result["test_loss"],
            "test_acc": result["test_acc"],
            "best_epoch": result["best_epoch"],
            "num_params": result["num_params"],
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
            f"Params: {metric['num_params']:,} | "
            f"Duration: {metric['duration']:.2f}s ({metric['duration']/60:.2f}min)"
        )

    dataset = cfg["data"]["dataset"]
    result_dir = Path(cfg["train"].get("result_dir", "results")) / dataset
    save_path = result_dir / "final_metrics.json"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main("configs.yaml")
