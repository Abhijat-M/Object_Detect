import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.voc import VOCDataset
from models.frcnn import build_model
from engine.train import train_one_epoch



DATASET_ROOT = r"C:/Users/Wizard/Documents/Object_Detect/PASCAL_VOC/VOC2012_train_val/VOC2012_train_val"

CLASSES = ["person", "car", "dog", "bicycle"]
NUM_CLASSES = len(CLASSES) + 1

EPOCHS = 40
BATCH_SIZE = 2
LR = 0.005


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    transform = transforms.ToTensor()

    dataset = VOCDataset(
        root=DATASET_ROOT,
        split="train",
        classes=CLASSES,
        transforms=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = build_model(NUM_CLASSES).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=0.0005
    )

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

        torch.save(model.state_dict(), "checkpoints/last.pth")


if __name__ == "__main__":
    main()
