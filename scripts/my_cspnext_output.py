from pathlib import Path

import fire
import torch

from train_rtmdet import CSPNeXtTiny, RotatedRTMDet


def main(
    ckpt: Path,
    sample_input: Path = Path("./random_sample_input.pt"),
    out_path: Path = Path("./my_output.pt"),
):
    model = RotatedRTMDet(
        backbone=CSPNeXtTiny(),
    )
    state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    x = torch.load(sample_input)
    model.eval()
    with torch.inference_mode():
        out = model(x)

    torch.save(out, out_path)


if __name__ == "__main__":
    fire.Fire(main)
