import os

def split_input_mem(input_file="input.mem", output_dir="split_mem_files"):
    """
    Splits a single 784-entry (28x28) input.mem file (signed Q8.8 hex values)
    into 28 separate files: row0.mem ... row27.mem

    Each output file will contain 28 hex values corresponding to one image row.
    Files are saved under 'split_mem_files/' by default.
    """

    # Read all hex values (strip blank lines)
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Sanity check
    if len(lines) != 784:
        raise ValueError(f"❌ Expected 784 values (28x28 MNIST), got {len(lines)}")

    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Split and write
    for r in range(28):
        start = r * 28
        end = start + 28
        row_vals = lines[start:end]
        out_path = os.path.join(output_dir, f"row{r+1}.mem")

        with open(out_path, "w") as out_f:
            out_f.write("\n".join(row_vals) + "\n")

    print(f"✅ Successfully split '{input_file}' into 28 row files inside '{output_dir}/'")
    print("   Each file has 28 signed Q8.8 hex values (row-major order).")

if __name__ == "__main__":
    split_input_mem("input.mem", "split_mem_files")
