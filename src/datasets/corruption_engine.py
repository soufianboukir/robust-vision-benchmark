import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io
import random


# apply gaussian noise
def add_gaussian_noise(image, severity): # image is a tensor of shape(Channel, H, W) with values [0,1]
    std_levels = [0.0, 0.02, 0.04, 0.06, 0.1] # standard deviation levels from 0 to 0.20 more noise
    std = std_levels[severity] # get the standard deviation value using sevrity level
    if std == 0.0: # if the standard deviation equals 0, we return the copy of the image
        # a = image
        # b = image
        # Now:
        # a and b are the same object
        # modifying one modifies both
        return image.clone() # key idea: “no change” ≠ “same object”
    noise = torch.randn_like(image) * std # generates a noise tensor with X follows a normal distribution (0,1), multiplying this noise with std to scale it
    return torch.clamp(image + noise, 0.0, 1.0) # apply clamping to ensure that the values are in range (0,1)



# build a 2D gaussian kernel
def _gaussian_kernel(kernel_size, sigma):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2 # generates coords (array of integers) based on the kernel size if it's 5 then coords = [0, 1, 2, 3, 4] - 2 = [-2, -1 , 0, 1, 2]
    # I used the torch.float32 above to ensure that the results of g bottom will be floats not ints cause sometimes int/int = 0, ex: 1/2 = 0: in many old math libraries
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)) # apply the gaussian formula to get [0, 1, 2, 3, 4] the importance of each pixel
    g /= g.sum() # normalization (make sure that the values are sum to 1), new_pixel = old_pixel / sum_pixels
    # g = [a, b, c]
    # outer(g, g) =
    # a*a   a*b   a*c
    # b*a   b*b   b*c
    # c*a   c*b   c*c
    kernel = g.outer(g) # apply the outer product above

    return kernel.unsqueeze(0).unsqueeze(0) # add 2 other dimensions to the kernel so it will be (out_channels, in_channels, h, w)


# apply blur
def apply_blur(image, severity):
    params = [
        (1, 0.0),   # level 0 – clean
        (3, 0.4),   # level 1 – slight
        (5, 0.8),   # level 2 – moderate
        (7, 1.2),   # level 3 – strong
        (9, 1.6),   # level 4 – extreme
    ]
    k_size, sigma = params[severity] # get the kernel_size and the sigma value vased on severity level
    if sigma == 0.0: # return the copy of the image if severity level is 0
        return image.clone()

    C = image.shape[0] # get the number of channels, for CIFAR dataset, C = 3
    kernel = _gaussian_kernel(k_size, sigma).repeat(C, 1, 1, 1)  # get the gaussian kernel, shape before repeat: (1, 1, k, k), 
    # after repeat = (3, 1, k, k), this means repeat filters C times, 1,1,1 means do not change other parameters
    pad = k_size // 2 # add padding to avoid image shrinking after applying kernels
    img4d = image.unsqueeze(0)                         # (1, C, H, W) cause Conve2d expects an input that has batch_size
    blurred = F.conv2d(img4d, kernel, padding=pad, groups=C) # apply each kernel to each channel independantely
    return torch.clamp(blurred.squeeze(0), 0.0, 1.0) # make sure that the values are in range 0,1 and remove the batch size dimension


# apply Occlusion: Random black squares on image
def apply_occlusion(image, severity):
    size_levels = [0, 2, 4, 8, 12]
    size = size_levels[severity] # get the block size based on severity level
    if size == 0: # return a clone of the image if severity is 0
        return image.clone()

    img = image.clone() # copy the image to prevent modify on the original one
    _, h, w = img.shape # get the w,h of the image, _ to ignore image channel
    x = random.randint(0, max(0, w - size - 1)) # random position of x
    y = random.randint(0, max(0, h - size - 1)) # random position of y
    img[:, y:y + size, x:x + size] = 0.0 # : apply this for all channels, from y to y + size, and from x to x + size, pixels set to 0
    return img


# Apply image compression
def apply_jpeg_compression(image, severity):
    quality_levels = [95, 85, 60, 35, 20]
    quality = quality_levels[severity] # get the compression level based on severity

    # tensor (C, H, W) float [0,1] → PIL uint8
    # convert a pytorch image to numpy image
    # PyTorch format, permute work:
        # (C, H, W)
        # PIL/NumPy format:
        # (H, W, C)
    # numpy: convert tensor to numpy array with values from 0 to 255
    # clip(255), ensures that all values are [0,255], astype(uint), make sure that the values are all integers
    np_img = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img) # convert to PIL image

    # image compression
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")

    out = torch.from_numpy(np.array(decoded)).permute(2, 0, 1).float() / 255.0 # convert it again to pytorch format & devide by 255 to make sure values are in [0,1]
    return out


# brightness/ contrast: Simulates lighting conditions
def apply_brightness_contrast(
    image,
    severity,
    mode: str = "brightness",
):
    brightness_deltas = [0.0, 0.07, 0.14, 0.21, 0.28]
    contrast_factors  = [1.0,0.85,0.75,0.65,0.55]

    img = image.clone()
    if mode == "brightness":
        delta = brightness_deltas[severity] # get the brightness delta
        return torch.clamp(img + delta, 0.0, 1.0) # we want the image to be more lighter
    elif mode == "contrast":
        factor = contrast_factors[severity] # get the contrast factor
        return torch.clamp((img - 0.5) * factor + 0.5, 0.0, 1.0) # we want everything moves toward 0.5 the middle(gray) and keep values in range [0,1]


# random rotation: Image augmentation
def apply_rotation(image, severity):
    max_angle_deg = [0, 10, 20, 30, 45]
    angle_deg = max_angle_deg[severity]  # FIX: exact value, no randomness

    if angle_deg == 0:
        return image.clone()

    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    theta = torch.tensor([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
    ], dtype=torch.float32).unsqueeze(0)

    img4d = image.unsqueeze(0)
    grid = F.affine_grid(theta, img4d.size(), align_corners=False)

    rotated = F.grid_sample(
        img4d,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )

    return rotated.squeeze(0)




CORRUPTION_TYPES = [
    "gaussian_noise",
    "blur",
    "occlusion",
    "jpeg_compression",
    "brightness",
    "contrast",
    "rotation",
]

# apply a corruption on a single image
def apply_corruption(
    image,
    corruption_type,
    severity,
):
    assert 0 <= severity <= 4, "severity must be 0-4" # assert used to ensure that the condition is True, if it's false stop the program
    # A lambda is a short, anonymous function.
    dispatch = {
        "gaussian_noise":  add_gaussian_noise,
        "blur":            apply_blur,
        "occlusion":       apply_occlusion,
        "jpeg_compression": apply_jpeg_compression,
        "brightness":      lambda img, sev: apply_brightness_contrast(img, sev, "brightness"),
        "contrast":        lambda img, sev: apply_brightness_contrast(img, sev, "contrast"),
        "rotation":        apply_rotation,
    }
    return dispatch[corruption_type](image, severity)


# appply corruption on a batch of images
def apply_corruption_batch(
    images,
    corruption_type,
    severity,
):
    return torch.stack([
        apply_corruption(img, corruption_type, severity)
        for img in images
    ])