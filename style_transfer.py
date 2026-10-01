"""Neural style transfer (Gatys et al., 2016), from Style_Transfer.ipynb.

Same method as the notebook: VGG19 features, content from conv4_2, style from Gram
matrices of five conv layers, and Adam optimizing the image itself. The only changes
are a higher learning rate and fewer steps so it finishes in minutes in a GUI,
plus a callback so the app can show progress.
"""
import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# conv layer index in vgg19.features -> name
LAYERS = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "21": "conv4_2",  # content representation
    "28": "conv5_1",
}

# Early layers carry fine texture (brushstrokes), later ones larger structure
STYLE_WEIGHTS = {
    "conv1_1": 1.0,
    "conv2_1": 0.75,
    "conv3_1": 0.2,
    "conv4_1": 0.2,
    "conv5_1": 0.2,
}


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vgg(device: torch.device) -> torch.nn.Module:
    """Pretrained VGG19 feature extractor with frozen weights.

    Memory-lean so it fits on small hosts like Streamlit Community Cloud. The full
    checkpoint is ~550 MB, almost all of it the classifier we don't use. The first time,
    it's loaded once, just the convolutional weights the losses need (through conv5_1)
    are saved to a small file (~80 MB), and the rest is freed. After that only the small
    file is ever read. The network is also cut after conv5_1, which skips unused layers.
    """
    # Keep layers through conv5_1 *and the ReLU after it*: VGG's ReLUs are in-place, so in
    # the original notebook that ReLU also rewrites the stored conv5_1 features.
    last = max(int(i) for i in LAYERS) + 2          # layers 0..29
    checkpoints = os.path.join(torch.hub.get_dir(), "checkpoints")
    small = os.path.join(checkpoints, f"vgg19-features-{last}.pth")

    if not os.path.exists(small):
        weights = models.VGG19_Weights.IMAGENET1K_V1
        full = os.path.join(checkpoints, os.path.basename(weights.url))
        if not os.path.exists(full):
            os.makedirs(checkpoints, exist_ok=True)
            torch.hub.download_url_to_file(weights.url, full)
        state = torch.load(full, map_location="cpu", weights_only=True)
        features = {k[len("features."):]: v.clone() for k, v in state.items()
                    if k.startswith("features.") and int(k.split(".")[1]) < last}
        del state
        torch.save(features, small)
        os.remove(full)                             # the big file isn't needed any more
        del features

    vgg = models.vgg19(weights=None).features[:last]
    vgg.load_state_dict(torch.load(small, map_location="cpu", weights_only=True))
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg.to(device).eval()


def to_tensor(image: Image.Image, max_size: int, shape: Optional[tuple] = None) -> torch.Tensor:
    """Resize (longest side <= max_size, or to `shape`), normalize, add a batch dim."""
    image = image.convert("RGB")
    size = shape if shape is not None else min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image)[:3].unsqueeze(0)


def to_image(tensor: torch.Tensor) -> Image.Image:
    """Undo normalization and turn a (1, 3, H, W) tensor into a PIL image."""
    array = tensor.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
    array = (array * np.array(STD) + np.array(MEAN)).clip(0, 1)
    return Image.fromarray((array * 255).astype(np.uint8))


def get_features(image: torch.Tensor, model: torch.nn.Module) -> dict:
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in LAYERS:
            features[LAYERS[name]] = x
    return features


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)
    return torch.mm(tensor, tensor.t())


def stylize(
    content_img: Image.Image,
    style_img: Image.Image,
    vgg: torch.nn.Module,
    device: torch.device,
    max_size: int = 384,
    steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    lr: float = 0.02,
    on_progress: Optional[Callable[[int, float, Optional[Image.Image]], None]] = None,
    preview_every: int = 25,
) -> Image.Image:
    """Repaint `content_img` in the style of `style_img`. Returns the result image."""
    content = to_tensor(content_img, max_size).to(device)
    style = to_tensor(style_img, max_size, shape=content.shape[-2:]).to(device)

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_WEIGHTS}

    # Start from the content photo and nudge its pixels toward the style
    target = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=lr)

    for step in range(1, steps + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

        style_loss = 0.0
        for layer, weight in STYLE_WEIGHTS.items():
            feature = target_features[layer]
            _, d, h, w = feature.shape
            layer_loss = weight * torch.mean((gram_matrix(feature) - style_grams[layer]) ** 2)
            style_loss += layer_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if on_progress:
            preview = to_image(target) if (step % preview_every == 0 or step == steps) else None
            on_progress(step, total_loss.item(), preview)

    return to_image(target)
