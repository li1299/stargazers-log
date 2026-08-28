"""
训练脚本 — 宠物猫品种分类
支持 VGG16 / ResNet50 / MobileNetV2
数据集目录结构:
    data/
      train/
        阿比西尼亚猫/  img1.jpg ...
        孟加拉猫/      ...
        ...
      val/
        阿比西尼亚猫/  ...
        ...

用法:
    python train.py --model ResNet50 --epochs 30 --batch_size 32 --data_dir ./data
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score
)
import seaborn as sns

# ── 导入主程序的模型构建函数 ────────────────────────────────────────
from cat_classifier import build_model, build_transform, CAT_BREEDS, NUM_CLASSES


def get_loaders(data_dir: str, batch_size: int):
    train_tf = build_transform(augment=True)
    val_tf   = build_transform(augment=False)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"训练集: {len(train_ds)} 张  验证集: {len(val_ds)} 张")
    print(f"类别数: {len(train_ds.classes)}")
    return train_loader, val_loader, train_ds.classes


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    avg_loss = total_loss / total
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_accs, "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, val_accs,   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves.png"), dpi=150)
    plt.close()
    print(f"曲线图已保存到 {save_dir}/curves.png")


def plot_confusion_matrix(labels, preds, class_names, save_dir, model_name):
    n = min(len(class_names), 15)  # 最多显示15类，避免过密
    cm = confusion_matrix(labels, preds)[:n, :n]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names[:n],
                yticklabels=class_names[:n],
                linewidths=0.5)
    plt.title(f"Confusion Matrix — {model_name} (前{n}类)")
    plt.ylabel("真实类别")
    plt.xlabel("预测类别")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存到 {path}")


def measure_inference_speed(model, device, n_runs=50):
    """测量 1080P 图片推理速度（毫秒）"""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    # 预热
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="宠物猫品种分类训练脚本")
    parser.add_argument("--model",      type=str,   default="ResNet50",
                        choices=["VGG16", "ResNet50", "MobileNetV2"])
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--data_dir",   type=str,   default=r"D:\workbody\2026-06-12-18-46-29\oxford-pet-12cats-split")
    parser.add_argument("--save_dir",   type=str,   default="./outputs")
    parser.add_argument("--compare",    action="store_true",
                        help="对比三种模型（忽略 --model 参数）")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    models_to_train = (
        ["VGG16", "ResNet50", "MobileNetV2"] if args.compare else [args.model]
    )

    all_results = {}

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"  训练模型: {model_name}")
        print(f"{'='*60}")

        model_save_dir = os.path.join(args.save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        train_loader, val_loader, class_names = get_loaders(
            args.data_dir, args.batch_size)

        model = build_model(model_name, len(class_names))
        model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []
        best_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler)
            vl_loss, vl_acc, _, _ = evaluate(
                model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            train_losses.append(tr_loss); val_losses.append(vl_loss)
            train_accs.append(tr_acc);   val_accs.append(vl_acc)

            print(f"  Epoch [{epoch:>3}/{args.epochs}] "
                  f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} "
                  f"ValLoss={vl_loss:.4f} ValAcc={vl_acc:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f} "
                  f"({elapsed:.1f}s)")

            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(model.state_dict(),
                           os.path.join(model_save_dir, "best.pth"))
                print(f"    ✅  最优模型已保存 (val_acc={best_acc:.4f})")

        # ── 最终评估 ────────────────────────────────────────────────
        _, final_acc, all_labels, all_preds = evaluate(
            model, val_loader, criterion, device)

        prec   = precision_score(all_labels, all_preds, average="weighted",
                                  zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted",
                              zero_division=0)
        speed  = measure_inference_speed(model, device)

        print(f"\n  {model_name} 最终评估:")
        print(f"    准确率: {final_acc*100:.2f}%")
        print(f"    精确率: {prec*100:.2f}%")
        print(f"    召回率: {recall*100:.2f}%")
        print(f"    推理速度: {speed:.1f} ms/张")

        all_results[model_name] = {
            "accuracy":  round(final_acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall":    round(recall * 100, 2),
            "speed_ms":  round(speed, 1),
        }

        # 保存曲线 + 混淆矩阵
        plot_curves(train_losses, val_losses, train_accs, val_accs, model_save_dir)
        plot_confusion_matrix(all_labels, all_preds, class_names,
                              model_save_dir, model_name)

        # 保存分类报告
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names[:len(np.unique(all_labels))],
            zero_division=0
        )
        with open(os.path.join(model_save_dir, "report.txt"), "w",
                  encoding="utf-8") as f:
            f.write(report)

    # ── 汇总对比 ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  三模型对比汇总")
    print(f"{'='*60}")
    print(f"  {'模型':<14} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'速度(ms)':>10}")
    print(f"  {'-'*52}")
    for m, r in all_results.items():
        print(f"  {m:<14} {r['accuracy']:>7.2f}% {r['precision']:>7.2f}% "
              f"{r['recall']:>7.2f}% {r['speed_ms']:>9.1f}")

    with open(os.path.join(args.save_dir, "comparison.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存到 {args.save_dir}/comparison.json")


if __name__ == "__main__":
    main()
