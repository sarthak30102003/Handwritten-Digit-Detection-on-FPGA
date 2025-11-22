import os
import numpy as np

# =========================================================
# Q8.8 Conversion: Float → 16-bit Signed Hex
# =========================================================
def float_to_q8_8_hex(value):
    """
    Convert a Q8.8 signed decimal float to 16-bit hexadecimal representation.
    Q8.8: 1 sign bit, 7 integer bits, 8 fractional bits.
    """
    # Scale and round
    scaled = int(np.round(value * (1 << 8)))

    # Clamp to signed 16-bit range (-32768 to 32767)
    if scaled > 32767:
        scaled = 32767
    elif scaled < -32768:
        scaled = -32768

    # Convert to two's complement if negative
    if scaled < 0:
        scaled = (1 << 16) + scaled

    # Format as 4-digit uppercase hex (zero-padded)
    return f"{scaled:04X}"


# =========================================================
# Convert Whole File
# =========================================================
def convert_q8_8_file_to_hex(input_path, output_path):
    """
    Read float Q8.8 values from a .txt file, convert each to 16-bit signed hex,
    and save as a .mem file (one hex word per line).
    """
    # Read float values
    data = np.loadtxt(input_path, dtype=np.float64).flatten()

    # Convert to hex
    hex_data = [float_to_q8_8_hex(v) for v in data]

    # Save
    with open(output_path, "w") as f:
        for h in hex_data:
            f.write(h + "\n")

    print(f"✅ Conversion complete!")
    print(f"   Input : {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Total values converted: {len(hex_data)}")


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    input_file = "input.txt"   
    output_file = "input.mem"
    convert_q8_8_file_to_hex(input_file, output_file)
