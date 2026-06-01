import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from model_resnet34_unet import ResNet34UNet


@torch.no_grad()
def predict_single_image(
    image_path,
    checkpoint_path,
    save_path,
    image_size=256,
    threshold=0.5,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34UNet(
        num_classes=1,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    x = TF.resize(image, [image_size, image_size])
    x = TF.to_tensor(x)

    x = TF.normalize(
        x,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    x = x.unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.sigmoid(logits)

    mask = (prob > threshold).float()
    mask = mask.squeeze().cpu().numpy()

    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)

    mask = mask.resize(original_size, resample=Image.NEAREST)
    mask.save(save_path)

    print(f"Saved prediction mask to: {save_path}")


if __name__ == "__main__":
    predict_single_image(
        image_path="data/oxford_102_flower/images/image_00001.jpg",
        checkpoint_path="checkpoints/best_resnet34_unet.pth",
        save_path="prediction.png",
        image_size=256
    )