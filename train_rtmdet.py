from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from rtmdet.dataset import DOTADataset, dota_collate_fn, OrientedBoundingBoxBatch
from rtmdet.model import RotatedRTMDet
from rtmdet.loss import RotatedRTMDetLoss


def train(
    num_workers: int = 0, epochs: int = 10, image_size: tuple[int, int] = (512, 512)
):
    model = RotatedRTMDet(model_name="rtmdet-tiny")

    train_dataset = DOTADataset(
        Path("./data/dota128/"),
        split="train",
        transform=v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), v2.Resize(image_size)]
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

    criterion = RotatedRTMDetLoss(image_shape=image_size)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            batch: OrientedBoundingBoxBatch
            output = model(batch)
            loss = criterion(batch, output)

            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch}] | Loss [{loss.item()}]")


if __name__ == "__main__":
    train()
