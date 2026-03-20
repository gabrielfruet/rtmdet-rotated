from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from rtmdet.dataset import DOTADataset, dota_collate_fn, OrientedBoundingBoxBatch
from rtmdet.model import RotatedRTMDet


def train(num_workers: int = 0, epochs: int = 10):
    model = RotatedRTMDet(model_name="rtmdet_tiny")

    train_dataset = DOTADataset(
        Path("./data/dota128/"),
        split="train",
        transform=v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), v2.Resize((512, 512))]
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=dota_collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(epochs):
        for batch in train_dataloader:
            batch: OrientedBoundingBoxBatch
            optimizer.zero_grad()

            output = model(batch)


if __name__ == "__main__":
    train()
