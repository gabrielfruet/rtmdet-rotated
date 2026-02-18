import torch

from rtmdet.model import CSPNeXtTiny, RotatedRTMDet

if __name__ == "__main__":
    import cv2
    import numpy as np

    model = RotatedRTMDet(CSPNeXtTiny())
    ckpt = torch.load(
        "./cspnext-tiny_imagenet_600e.pth",
        map_location="cpu",
    )
    missing_keys, unexpected_keys = model.load_state_dict(
        ckpt["state_dict"], strict=False
    )
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    image = cv2.imread("/home/fruet/Pictures/profpic.png")
    assert image is not None
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    k = cv2.waitKey(0)

    image = cv2.resize(image, (224, 224))
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    outs = model(x)[-3]

    fmap_out = (
        torch.norm(outs, dim=1, keepdim=True).squeeze(0).squeeze(0).detach().numpy()
    )
    fmap_out = cv2.normalize(fmap_out, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    fmap_out = cv2.applyColorMap(fmap_out, cv2.COLORMAP_JET)

    cv2.imshow("Image", fmap_out)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
