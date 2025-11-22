import numpy as np
import os

# =========================================================
# Global Config
# =========================================================
class Config:
    model_dir = "model_params"
    input_path = "input.txt"
    feature_dir = "layer_outputs"


# =========================================================
# Architecture Parameters (must match training)
# =========================================================
class ModelConfig:
    input_size = (28, 28)
    row = {"kernel_size": (1, 8), "stride": (1, 2)}
    col = {"kernel_size": (8, 1), "stride": (2, 1)}
    fc1_nodes = 32
    fc2_nodes = 10


# =========================================================
# Q8.8 Helpers
# =========================================================
def float_to_q8_8(x):
    scale = 2 ** 8
    x_clamped = np.clip(x, -128.0, 127.99609375)
    return np.round(x_clamped * scale) / scale


def normalize_minus1_plus1_q8_8(x):
    x_norm = 2.0 * x - 1.0
    return float_to_q8_8(x_norm)


# =========================================================
# Utility
# =========================================================
def load_txt(path):
    data = np.loadtxt(path, dtype=np.float32)
    return float_to_q8_8(data)


def load_and_reshape(path, out_features, in_features):
    data = load_txt(path)
    return data.reshape(out_features, in_features)


def save_featuremap(name, data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, f"{name}.txt"), float_to_q8_8(data), fmt="%.8f")


# =========================================================
# Core Operations (correlation)
# =========================================================
def conv2d_rowwise(input_img, kernel, stride=(1, 2)):
    H, W = input_img.shape
    kh, kw = kernel.shape
    sh, sw = stride
    out_h = (H - kh) // sh + 1 if kh > 1 else H
    out_w = (W - kw) // sw + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            region = input_img[r:r + kh, c * sw:c * sw + kw]
            mac = np.sum(region * kernel)
            output[r, c] = float_to_q8_8(mac)
    return output


def conv2d_colwise(input_img, kernel, stride=(2, 1)):
    H, W = input_img.shape
    kh, kw = kernel.shape
    sh, sw = stride
    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1 if kw > 1 else W
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            region = input_img[r * sh:r * sh + kh, c:c + kw]
            mac = np.sum(region * kernel)
            output[r, c] = float_to_q8_8(mac)
    return output


def linear(x, weight, bias):
    return float_to_q8_8(np.dot(weight, x) + bias)


# =========================================================
# Inference Pipeline (column-major flatten)
# =========================================================
def main():
    cfg = Config()
    mcfg = ModelConfig()

    row_kh, row_kw = mcfg.row["kernel_size"]
    col_kh, col_kw = mcfg.col["kernel_size"]

    # --- Load kernels (no flip) ---
    row_conv_w = load_txt(os.path.join(cfg.model_dir, "row_conv_weights.txt")).reshape(row_kh, row_kw)
    col_conv_w = load_txt(os.path.join(cfg.model_dir, "col_conv_weights.txt")).reshape(col_kh, col_kw)

    H, W = mcfg.input_size
    _, sw = mcfg.row["stride"]
    sh, _ = mcfg.col["stride"]
    out_row_w = (W - row_kw) // sw + 1
    out_col_h = (H - col_kh) // sh + 1
    flattened_dim = out_col_h * out_row_w

    fc1_w = load_and_reshape(os.path.join(cfg.model_dir, "fc1_weights.txt"),
                             mcfg.fc1_nodes, flattened_dim)
    fc1_b = load_txt(os.path.join(cfg.model_dir, "fc1_bias.txt"))
    fc2_w = load_and_reshape(os.path.join(cfg.model_dir, "fc2_weights.txt"),
                             mcfg.fc2_nodes, mcfg.fc1_nodes)
    fc2_b = load_txt(os.path.join(cfg.model_dir, "fc2_bias.txt"))

    # --- Load input ---
    img = float_to_q8_8(np.loadtxt(cfg.input_path, dtype=np.float32))

    print("Running inference pipeline...\n")

    x = conv2d_rowwise(img, row_conv_w, stride=mcfg.row["stride"])
    save_featuremap("row_conv_out", x, cfg.feature_dir)

    x = conv2d_colwise(x, col_conv_w, stride=mcfg.col["stride"])
    save_featuremap("col_conv_out", x, cfg.feature_dir)

    if x.ndim == 2:
        x = np.transpose(x, (1, 0)).flatten()
    else:
        x = np.concatenate([np.transpose(ch, (1, 0)).flatten() for ch in x], axis=0)
    save_featuremap("flatten_out", x, cfg.feature_dir)

    x = linear(x, fc1_w, fc1_b)
    save_featuremap("fc1_out", x, cfg.feature_dir)

    logits = linear(x, fc2_w, fc2_b)
    save_featuremap("fc2_out_logits", logits, cfg.feature_dir)

    pred = np.argmax(logits)
    print(f"\n✅ Inference Complete\nPredicted Digit: {pred}")
    print("Logits (Q8.8):", np.round(logits, 6))


if __name__ == "__main__":
    main()
