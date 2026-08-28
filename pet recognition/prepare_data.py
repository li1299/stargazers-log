"""
数据集准备辅助脚本
从 Oxford-IIIT Pet Dataset (或本地图片文件夹) 自动生成 train/val 目录结构

用法:
    # 从 Oxford Pets 官方压缩包
    python prepare_data.py --source oxford --images_dir ./images --annotations_dir ./annotations

    # 从自定义目录 (每个子目录名为类别名)
    python prepare_data.py --source custom --raw_dir ./raw_images --output_dir ./data --val_ratio 0.15
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image


def split_custom(raw_dir: str, output_dir: str, val_ratio: float = 0.15):
    """
    将 raw_dir/<类别>/*.jpg 格式的数据集按比例分割为 train/val
    """
    raw = Path(raw_dir)
    out = Path(output_dir)
    train_dir = out / "train"
    val_dir   = out / "val"

    classes = sorted([d.name for d in raw.iterdir() if d.is_dir()])
    print(f"发现 {len(classes)} 个类别")

    for cls in classes:
        imgs = list((raw / cls).glob("*.*"))
        imgs = [p for p in imgs
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        val_imgs   = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for phase, lst in [("train", train_imgs), ("val", val_imgs)]:
            dest = out / phase / cls
            dest.mkdir(parents=True, exist_ok=True)
            for p in lst:
                shutil.copy2(p, dest / p.name)

        print(f"  {cls:<30} train={len(train_imgs):>4}  val={len(val_imgs):>4}")

    print(f"\n数据集已准备好: {output_dir}")


def prepare_oxford(images_dir: str, annotations_dir: str, output_dir: str,
                   val_ratio: float = 0.15):
    """
    解析 Oxford-IIIT Pets 的 list.txt 标注，按品种建目录
    只保留猫 (class_id >= 13 or species==2，取决于版本)
    """
    list_file = Path(annotations_dir) / "list.txt"
    images    = Path(images_dir)
    out       = Path(output_dir)

    samples: dict[str, list] = {}
    with open(list_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            img_name, class_id, species, _ = (
                parts[0], int(parts[1]), int(parts[2]), parts[3])
            if species != 2:   # 1=dog, 2=cat
                continue
            # 品种名 = img_name 去掉末尾 _NNN
            breed = "_".join(img_name.split("_")[:-1])
            samples.setdefault(breed, []).append(img_name)

    print(f"共找到 {len(samples)} 种猫品种")
    all_items = [(breed, name) for breed, names in samples.items()
                 for name in names]
    random.shuffle(all_items)

    for breed, name in all_items:
        src = images / f"{name}.jpg"
        if not src.exists():
            continue
        phase = "val" if random.random() < val_ratio else "train"
        dest  = out / phase / breed
        dest.mkdir(parents=True, exist_ok=True)
        try:
            # 验证图片合法性
            img = Image.open(src)
            img.verify()
            shutil.copy2(src, dest / f"{name}.jpg")
        except Exception:
            pass  # 跳过损坏图片

    print(f"数据集已准备好: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["oxford", "custom"], default="custom")
    parser.add_argument("--raw_dir",          default="./raw_images")
    parser.add_argument("--images_dir",       default="./images")
    parser.add_argument("--annotations_dir",  default="./annotations")
    parser.add_argument("--output_dir",       default="./data")
    parser.add_argument("--val_ratio",        type=float, default=0.15)
    args = parser.parse_args()

    random.seed(42)
    if args.source == "oxford":
        prepare_oxford(args.images_dir, args.annotations_dir,
                       args.output_dir, args.val_ratio)
    else:
        split_custom(args.raw_dir, args.output_dir, args.val_ratio)
