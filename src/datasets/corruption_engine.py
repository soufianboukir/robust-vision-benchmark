import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io
import random


# apply gaussian noise
def add_gaussian_noise(image, severity):
    std_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    std = std_levels[severity]
    if std == 0.0:
        return image.clone()
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0.0, 1.0)



def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Builds a 2-D Gaussian kernel as a 4-D conv weight tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)                                # (k, k)
    # shape: (out_ch, in_ch/groups, k, k) – used with groups=C for depthwise
    return kernel.unsqueeze(0).unsqueeze(0)


# apply blur
def apply_blur(image: torch.Tensor, severity: int) -> torch.Tensor:
    """
    Applies Gaussian blur with increasing kernel / sigma per severity level.
    severity 0 → identity, 4 → heavy blur.
    """
    params = [
        (1, 0.0),   # level 0 – clean
        (3, 0.5),   # level 1 – slight
        (5, 1.0),   # level 2 – moderate
        (7, 1.5),   # level 3 – strong
        (9, 2.0),   # level 4 – extreme
    ]
    k_size, sigma = params[severity]
    if sigma == 0.0:
        return image.clone()

    C = image.shape[0]
    kernel = _gaussian_kernel(k_size, sigma).repeat(C, 1, 1, 1)  # (C,1,k,k)
    pad = k_size // 2
    img4d = image.unsqueeze(0)                         # (1, C, H, W)
    blurred = F.conv2d(img4d, kernel, padding=pad, groups=C)
    return torch.clamp(blurred.squeeze(0), 0.0, 1.0)


# apply Occlusion: Random black squares on image
def apply_occlusion(image, severity):
    size_levels = [0, 4, 8, 12, 16]
    size = size_levels[severity]
    if size == 0:
        return image.clone()

    img = image.clone()
    _, h, w = img.shape
    x = random.randint(0, max(0, w - size - 1))
    y = random.randint(0, max(0, h - size - 1))
    img[:, y:y + size, x:x + size] = 0.0
    return img


# Apply image compression
def apply_jpeg_compression(image, severity):
    quality_levels = [95, 75, 50, 25, 10]
    quality = quality_levels[severity]

    # tensor (C, H, W) float [0,1] → PIL uint8
    np_img = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")

    out = torch.from_numpy(np.array(decoded)).permute(2, 0, 1).float() / 255.0
    return out


# brightness/ contrast: Simulates lighting conditions
def apply_brightness_contrast(
    image,
    severity,
    mode: str = "brightness",
):
    """
    Simulates lighting conditions.
    mode='brightness': additive shift (positive = lighter, negative = darker)
    mode='contrast':   multiplicative scaling around 0.5 mid-grey

    severity 0 → identity, higher → stronger shift.
    """
    brightness_deltas = [0.0, 0.08, 0.16, 0.24, 0.32]
    contrast_factors  = [1.0, 0.85, 0.70, 0.55, 0.40]

    img = image.clone()
    if mode == "brightness":
        delta = brightness_deltas[severity]
        return torch.clamp(img + delta, 0.0, 1.0)
    elif mode == "contrast":
        factor = contrast_factors[severity]
        return torch.clamp((img - 0.5) * factor + 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'brightness' or 'contrast'.")


# random rotation: Image augmentation
def apply_rotation(image: torch.Tensor, severity: int) -> torch.Tensor:
    """
    Applies a random rotation whose maximum angle scales with severity.
    Uses bilinear grid-sampling; corners are filled with 0 (black).

    severity 0 → 0°, 4 → up to ±45°.
    """
    max_angle_deg = [0, 10, 20, 30, 45]
    max_deg = max_angle_deg[severity]
    if max_deg == 0:
        return image.clone()

    angle_deg = random.uniform(-max_deg, max_deg)
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # Affine rotation matrix (2×3) for grid_sample
    theta = torch.tensor([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
    ], dtype=torch.float32).unsqueeze(0)   # (1, 2, 3)

    img4d = image.unsqueeze(0)             # (1, C, H, W)
    grid  = F.affine_grid(theta, img4d.size(), align_corners=False)
    rotated = F.grid_sample(img4d, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
    return torch.clamp(rotated.squeeze(0), 0.0, 1.0)


# ─────────────────────────────────────────────
#  Master dispatcher
# ─────────────────────────────────────────────
CORRUPTION_TYPES = [
    "gaussian_noise",
    "blur",
    "occlusion",
    "jpeg_compression",
    "brightness",
    "contrast",
    "rotation",
]


def apply_corruption(
    image,
    corruption_type,
    severity,
) -> torch.Tensor:
    """
    Apply a named corruption at a given severity level.

    Args:
        image          : Float tensor of shape (C, H, W) in [0, 1].
        corruption_type: One of CORRUPTION_TYPES.
        severity       : Integer in range [0, 4].
                         0 = clean / identity.
                         4 = extreme corruption.

    Returns:
        Corrupted float tensor of the same shape, values in [0, 1].
    """
    assert 0 <= severity <= 4, "severity must be 0–4"
    dispatch = {
        "gaussian_noise":  add_gaussian_noise,
        "blur":            apply_blur,
        "occlusion":       apply_occlusion,
        "jpeg_compression": apply_jpeg_compression,
        "brightness":      lambda img, sev: apply_brightness_contrast(img, sev, "brightness"),
        "contrast":        lambda img, sev: apply_brightness_contrast(img, sev, "contrast"),
        "rotation":        apply_rotation,
    }
    if corruption_type not in dispatch:
        raise ValueError(
            f"Unknown corruption '{corruption_type}'. "
            f"Choose from: {list(dispatch.keys())}"
        )
    return dispatch[corruption_type](image, severity)


# ─────────────────────────────────────────────
#  Batch helper  (useful in evaluation loops)
# ─────────────────────────────────────────────
def apply_corruption_batch(
    images: torch.Tensor,
    corruption_type: str,
    severity: int,
) -> torch.Tensor:
    """
    Apply the same corruption to every image in a batch.

    Args:
        images: Float tensor of shape (N, C, H, W) in [0, 1].

    Returns:
        Corrupted batch tensor of the same shape.
    """
    return torch.stack([
        apply_corruption(img, corruption_type, severity)
        for img in images
    ])