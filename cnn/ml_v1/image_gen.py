import os
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# =========================================================
# Q8.8 Helpers (signed)
# =========================================================
def float_to_q8_8(x):
    """Convert float array to signed Q8.8 fixed-point representation."""
    scale = 2 ** 8
    x_clamped = np.clip(x, -128.0, 127.99609375)
    return np.round(x_clamped * scale) / scale


def normalize_to_minus1_plus1_q8_8(x):
    """
    Normalize MNIST image (original [0,1]) to [-1, +1],
    then quantize to signed Q8.8.
        x_norm = 2x - 1
    """
    x_norm = 2.0 * x - 1.0
    return float_to_q8_8(x_norm)


# =========================================================
# Main Export Function
# =========================================================
def save_random_mnist_q8_8(output_dir=".", idx=None, seed=None):
    """
    Pick a random MNIST test image, normalize to [-1, +1],
    quantize to signed Q8.8, and export as:
      - input_q8_8.txt  (for FPGA .mem conversion)
      - mnist_img{idx}_label{label}.png  (visual reference)
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load MNIST test set (raw 0–1 images)
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Pick random image if not specified
    if idx is None:
        idx = random.randint(0, len(testset) - 1)

    img, label = testset[idx]  # img: tensor [1,28,28]
    img_np = img.squeeze(0).numpy()  # shape [28,28]

    # Normalize + quantize to Q8.8 signed
    img_q8_8 = normalize_to_minus1_plus1_q8_8(img_np)

    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save Quantized Input
    txt_path = os.path.join(output_dir, "input.txt")
    np.savetxt(txt_path, img_q8_8, fmt="%.8f")

    # Save reference grayscale PNG
    img_display = ((img_np * 255).astype(np.uint8))
    img_path = os.path.join(output_dir, f"mnist_img{idx}_label{label}.png")
    Image.fromarray(img_display).save(img_path)

    print(f"✅ Saved MNIST image #{idx} (label={label})")
    print(f"   - Quantized data: {txt_path}")
    print(f"   - Image preview : {img_path}")
    print(f"   - Shape         : {img_q8_8.shape}")
    print(f"   - Value range   : [{img_q8_8.min():.3f}, {img_q8_8.max():.3f}]")


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    save_random_mnist_q8_8(output_dir=".")
