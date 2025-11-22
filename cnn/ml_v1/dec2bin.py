import os
import numpy as np

# =========================================================
# Paths
# =========================================================
input_folder = "model_params"    # trained model .txt weights/biases
output_folder = "model_params_mem"     # output .mem folder
os.makedirs(output_folder, exist_ok=True)

# =========================================================
# Q8.8 Conversion Helper
# =========================================================
def float_to_q8_8_hex(value):
    """
    Convert a Q8.8 signed float to 16-bit two's complement hex (uppercase).
    Q8.8 = 1 sign bit + 7 integer bits + 8 fractional bits.
    """
    # Scale
    scaled = int(np.round(value * (1 << 8)))

    # Clamp
    if scaled > 32767:
        scaled = 32767
    elif scaled < -32768:
        scaled = -32768

    # Two's complement for negatives
    if scaled < 0:
        scaled = (1 << 16) + scaled

    return f"{scaled:04X}"  # 4 hex digits


# =========================================================
# Process All Files
# =========================================================
for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_folder, filename)

    # Load all float values
    try:
        data = np.loadtxt(input_path, dtype=np.float64).flatten()
    except Exception as e:
        print(f"⚠️ Skipped {filename} (load error: {e})")
        continue

    # Convert to hex
    hex_values = [float_to_q8_8_hex(v) for v in data]

    # Output file
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}.mem")

    # Write hex values
    with open(output_path, "w") as f:
        for val in hex_values:
            f.write(val + "\n")

    print(f"✅ Processed {filename} → {output_path}  ({len(hex_values)} values)")

print("\n🎯 All .txt files converted to Q8.8 16-bit .mem format successfully.")
print(f"📂 Output directory: {os.path.abspath(output_folder)}")
