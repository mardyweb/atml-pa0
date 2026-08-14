"""Shared utilities for ATML PA0."""
import json, random
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

RESULTS_DIR = Path("results"); FIGURES_DIR = Path("figures"); DATA_DIR = Path("data")
for d in (RESULTS_DIR, FIGURES_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

CIFAR10_CLASSES = ["plane","car","bird","cat","deer","dog",
                   "frog","horse","ship","truck"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """Call at the top of every notebook so runs are reproducible."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stratified_indices(targets, n_total, seed=42, num_classes=10):
    """Pick n_total indices with an equal number from each class."""
    targets = np.array(targets)
    per_class = n_total // num_classes
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(num_classes):
        pool = np.where(targets == c)[0]
        idx.extend(rng.choice(pool, per_class, replace=False))
    rng.shuffle(idx)
    return [int(i) for i in idx]


def subset_loaders(n_train=5000, n_val=1000, batch_size=64, seed=42, size=224):
    """CIFAR-10 subsets, resized + normalised for ImageNet-pretrained models.

    Same seed => same images every time, so experiments stay comparable.
    """
    tf = T.Compose([T.Resize(size), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_full = torchvision.datasets.CIFAR10(DATA_DIR, train=True,
                                              download=True, transform=tf)
    val_full   = torchvision.datasets.CIFAR10(DATA_DIR, train=False,
                                              download=True, transform=tf)
    train = Subset(train_full, _stratified_indices(train_full.targets, n_train, seed))
    val   = Subset(val_full,   _stratified_indices(val_full.targets,   n_val,   seed))
    return (DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=2),
            DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=2))


def save_results(name, data):
    """save_results('task1_baseline', {'train_loss': [...], 'val_acc': [...]})"""
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", path)


def save_fig(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved", path)
