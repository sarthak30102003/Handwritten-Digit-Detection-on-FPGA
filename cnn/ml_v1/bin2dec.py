import os
import numpy as np
from collections import Counter

# =========================================================
# Configuration
# =========================================================
original_folder = "model_params"   # float text files
fixed_folder    = "model_params_mem"     # generated .mem hex files

# =========================================================
# Signed Q8.8 Conversions
# =========================================================
def q8_8_hex_to_float(hex_str):
    """Convert 16-bit signed Q8.8 hex string to float."""
    val = int(hex_str, 16)
    # Two's complement for negatives
    if val & (1 << 15):
        val -= (1 << 16)
    return val / (1 << 8)


def float_to_q8_8_hex(x):
    """Convert float to signed Q8.8 hex string."""
    if x < -128 or x >= 128:
        raise ValueError(f"Value {x} out of signed Q8.8 range [-128, 127.996]")
    scaled = int(round(x * (1 << 8)))
    if scaled < 0:
        scaled += (1 << 16)
    return f"{scaled:04X}"

# =========================================================
# Helper: Mode Calculation
# =========================================================
def mode(lst):
    counts = Counter(lst)
    if not counts:
        return None
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return modes[0]

# =========================================================
# Main Verification
# =========================================================
all_errors = []
max_error_info = None
max_error_val = -1

for filename in os.listdir(original_folder):
    if not filename.endswith(".txt"):
        continue

    orig_path  = os.path.join(original_folder, filename)
    fixed_path = os.path.join(fixed_folder, os.path.splitext(filename)[0] + ".mem")

    if not os.path.exists(fixed_path):
        print(f"⚠️ Missing {fixed_path}, skipped.")
        continue

    # Load float values
    with open(orig_path, "r") as f:
        original_numbers = [float(x) for x in f.read().split() if x.strip()]

    # Load fixed-point hex values
    with open(fixed_path, "r") as f:
        fixed_hex_numbers = [line.strip() for line in f if line.strip()]

    # Convert hex → float
    recovered_numbers = [q8_8_hex_to_float(h) for h in fixed_hex_numbers]

    # Compute percentage errors
    for idx, (orig, rec, hex_val) in enumerate(zip(original_numbers, recovered_numbers, fixed_hex_numbers)):
        if orig != 0:
            err_percent = abs((rec - orig) / orig) * 100
            all_errors.append(err_percent)

            if err_percent > max_error_val:
                max_error_val = err_percent
                max_error_info = {
                    "file": filename,
                    "index": idx,
                    "original": orig,
                    "recovered": rec,
                    "error%": err_percent,
                    "hex": float_to_q8_8_hex(orig),
                }

# =========================================================
# Statistics
# =========================================================
if not all_errors:
    print("⚠️ No data verified.")
    exit()

errors_array = np.array(all_errors)
print(f"Max error%: {np.max(errors_array):.6f}")
print(f"Min error%: {np.min(errors_array):.6f}")
print(f"Average error%: {np.mean(errors_array):.6f}")
print(f"Median error%: {np.median(errors_array):.6f}")
print(f"Mode error%: {mode(list(np.round(errors_array,6))):.6f}")

# =========================================================
# Max Error Details
# =========================================================
if max_error_info:
    hex_val = max_error_info["hex"]
    dec_val = int(hex_val, 16)
    bin_val = bin(dec_val)[2:].zfill(16)

    print("\nNumber with maximum error:")
    print(f"File         : {max_error_info['file']}")
    print(f"Line Number  : {max_error_info['index'] + 1}")
    print(f"Original     : {max_error_info['original']}")
    print(f"Recovered    : {max_error_info['recovered']}")
    print(f"Error%       : {max_error_info['error%']:.6f}")
    print(f"Q8.8 hex     : {hex_val}")
    print(f"Binary (16-bit signed): {bin_val}")
