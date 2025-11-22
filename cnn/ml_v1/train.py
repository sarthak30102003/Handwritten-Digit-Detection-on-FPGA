import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from torchsummary import summary

# =========================================================
# Global Configuration & Hyperparameters
# =========================================================
class Config:
    batch_size = 64
    test_batch_size = 1000
    epochs = 25
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "model_params"
    model_path = os.path.join(save_dir, "rowcol_cnn_model.pth")
    os.makedirs(save_dir, exist_ok=True)
    seed = 42


# =========================================================
# Model Architecture Configuration
# =========================================================
class ModelConfig:
    input_size = (28, 28)

    row = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": (1, 8),
        "stride": (1, 2),
        "bias": False,
    }

    col = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": (8, 1),
        "stride": (2, 1),
        "bias": False,
    }

    fc1_nodes = 32
    fc2_nodes = 10


# =========================================================
# Quantization Helpers
# =========================================================
def float_to_q8_8_signed_tensor(x: torch.Tensor):
    scale = 2 ** 8
    x_clamped = torch.clamp(x, -128.0, 127.99609375)
    x_q = torch.round(x_clamped * scale) / scale
    return x_q


def normalize_minus1_plus1(x: torch.Tensor):
    x_norm = 2.0 * x - 1.0
    return float_to_q8_8_signed_tensor(x_norm)


def quantize_model_signed_q8_8(model: nn.Module):
    for param in model.parameters():
        param.data = float_to_q8_8_signed_tensor(param.data)


# =========================================================
# CNN Model Definition (non-flipped kernels)
# =========================================================
class RowColCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(RowColCNN, self).__init__()
        self.cfg = cfg

        # Convolutions (PyTorch Conv2d already uses correlation semantics)
        self.row_conv = nn.Conv2d(**cfg.row)
        self.col_conv = nn.Conv2d(**cfg.col)

        # Compute flattened dimension dynamically
        dummy = torch.zeros(1, cfg.row["in_channels"], *cfg.input_size)
        with torch.no_grad():
            out = self.col_conv(self.row_conv(dummy))
        flattened_dim = out.numel()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_dim, cfg.fc1_nodes)
        self.fc2 = nn.Linear(cfg.fc1_nodes, cfg.fc2_nodes)

    def forward(self, x):
        x = self.row_conv(x)
        x = self.col_conv(x)
        # Column-major flatten
        x = x.permute(0, 1, 3, 2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# =========================================================
# Data Loading
# =========================================================
def get_dataloaders(config: Config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: normalize_minus1_plus1(x))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
    return train_loader, test_loader


# =========================================================
# Export Model Parameters for FPGA (non-flipped)
# =========================================================
def save_to_txt(filename, array):
    np.savetxt(filename, array, fmt="%.8f")

def export_model_params(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def save_layer(layer, name):
        w = layer.weight.data.detach().cpu().numpy()  # (out,in,kh,kw)
        w = np.squeeze(w)  # -> (kh,kw) for single-channel
        save_to_txt(os.path.join(save_dir, f"{name}_weights.txt"), w)
        if layer.bias is not None:
            b = layer.bias.data.detach().cpu().numpy()
            save_to_txt(os.path.join(save_dir, f"{name}_bias.txt"), b)

    save_layer(model.row_conv, "row_conv")
    save_layer(model.col_conv, "col_conv")
    save_layer(model.fc1, "fc1")
    save_layer(model.fc2, "fc2")
    print(f"\n✅ All weights and biases saved to '{save_dir}/'.")


# =========================================================
# Evaluation
# =========================================================
def evaluate(model, device, data_loader):
    model.eval()
    correct, total, loss_total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return loss_total / len(data_loader), 100 * correct / total


# =========================================================
# Training Loop
# =========================================================
def main():
    torch.manual_seed(Config.seed)
    train_loader, test_loader = get_dataloaders(Config)

    model = RowColCNN(ModelConfig).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    print(f"Training on device: {Config.device}\n")
    print(f"{'Epoch':^5} | {'Train Acc (%)':^13} | {'Train Loss':^10} | {'Val Acc (%)':^11} | {'Val Loss':^9}")
    print("-" * 65)

    for epoch in range(1, Config.epochs + 1):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(Config.device), labels.to(Config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        val_loss, val_acc = evaluate(model, Config.device, test_loader)
        print(f"{epoch:^5} | {train_acc:^13.2f} | {train_loss:^10.4f} | {val_acc:^11.2f} | {val_loss:^9.4f}")

    quantize_model_signed_q8_8(model)
    export_model_params(model, Config.save_dir)
    torch.save(model.state_dict(), Config.model_path)
    test_loss, test_acc = evaluate(model, Config.device, test_loader)
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")
    print("📘 Model Summary:")
    summary(model, input_size=(1, *ModelConfig.input_size))


if __name__ == "__main__":
    main()
