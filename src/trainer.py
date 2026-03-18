import time
from pathlib import Path

import torch


def train_one_epoch(model, optimizer, criterion, loader, device):
    """returns: avg_loss, avg_acc"""
    model.train()
    total_loss = 0
    total_correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc


def evaluate(model, criterion, loader, device):
    """returns: avg_loss, avg_acc"""
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)

            total_correct += (preds == yb).sum().item()
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc


def train(model, optimizer, criterion, train_loader, val_loader, device, epochs, ckpt_dir, save_name):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    format_n = len(str(epochs))
    best_val_acc = 0
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / save_name

    for epoch in range(epochs):
        start = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(start))

        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        duration = time.time() - start
        print(
            f"[{ts}] Epoch {epoch+1:0{format_n}d}/{epochs}\n"
            f"Train: [{train_loss:.4f}, {train_acc:.4f}] | "
            f"Val: [{val_loss:.4f}, {val_acc:.4f}] | "
            f"{duration:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best model saved at {ckpt_path} with val_acc: {best_val_acc:.4f}")

        print()

    return history
