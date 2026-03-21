from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim import lr_scheduler

from rtmdet.dataset import DOTADataset, dota_collate_fn, OrientedBoundingBoxBatch
from rtmdet.model import RotatedRTMDet
from rtmdet.loss import RotatedRTMDetLoss

torch.autograd.set_detect_anomaly(True)


def train(
    num_workers: int = 0,
    epochs: int = 10,
    image_size: tuple[int, int] = (512, 512),
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RotatedRTMDet(model_name="rtmdet-tiny")
    model.to(device)

    train_dataset = DOTADataset(
        Path("./data/dota128/"),
        split="train",
        transform=v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), v2.Resize(image_size)]
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=dota_collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    criterion = RotatedRTMDetLoss(image_shape=image_size)

    scheduler = lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=2),
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 2),
        ],
        milestones=[2],
    )

    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            batch: OrientedBoundingBoxBatch
            batch = batch.to_device(device)
            output = model(batch)
            loss = criterion(batch, output)

            loss.backward()
            optimizer.step()
            print(
                f"Epoch [{epoch}] | Loss [{loss.item()}] | Lr [{optimizer.param_groups[0]['lr']}]"
            )

        scheduler.step()


if __name__ == "__main__":
    train()
