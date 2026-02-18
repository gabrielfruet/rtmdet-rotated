from pathlib import Path

import fire
import torch

torch.manual_seed(42)


def main(outpath: Path = Path("random_sample_input.pt")):
    x = torch.randn(1, 3, 640, 640)
    torch.save(x, outpath)


if __name__ == "__main__":
    fire.Fire(main)
