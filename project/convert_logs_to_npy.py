import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import re
import sys
import re
import pandas as pd
# ======================================================
# === Step 1: Discover SRAM dataset ===================
# ======================================================

def discover_sram_dataset(root):
    def infer_manufacturer(fam):
        fam = fam.lower()
        if fam.startswith("input_as"):
            return "Alliance"
        elif fam.startswith("input_cy"):
            return "Cypress"
        elif fam.startswith("input_is"):
            return "ISSI"
        elif fam.startswith("input_idt"):
            return "IDT"
        elif fam.startswith("input_re") or fam.startswith("input_ren"):
            return "Renesas"
        else:
            return "Unknown"

    def collect_txt():
        folder = os.path.join(root, family)
        return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".log")]

    def extract_last_digits(fname):
        base = os.path.splitext(os.path.basename(fname))[0]
        m = re.search(r"(\d+)(?!.*\d)", base)
        return int(m.group(1)) if m else None

    def build_id_map(files):
        m = {}
        for path in files:
            idn = extract_last_digits(path)
            if idn is None:
                continue
            if idn in m:
                # deterministic: keep lexicographically smallest path
                m[idn] = min(m[idn], path)
            else:
                m[idn] = path
        return m

    pairs = []
    families = sorted([d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)) and d not in ('.', '..')])
    pairs = []
    for family in families:
        famPath = os.path.join(root, family)
        

        #gpuf_dirs = [d for d in subdirs if d.lower().startswith("gpuf")]
        #puc_dirs  = [d for d in subdirs if d.lower().startswith("puc")]

        if not famPath:
            print(f"[SKIP] {family}: missing input")
            continue

        logFiles = collect_txt()
        #pucFiles  = collect_txt(puc_dirs)

        logMap = build_id_map(logFiles)
        #pucMap  = build_id_map(pucFiles)

        common_ids = sorted(set(logMap.keys()))
        print(f"[{family}] log={len(logMap)}  paired={len(common_ids)}")

        for idn in common_ids:
            chip_id = f"{family}_id{idn:02d}"
            pairs.append({
                "family": family,
                "manufacturer": infer_manufacturer(family),
                "chip_id": chip_id,
                "log_path": logMap[idn],
            })
    
    pairs = sorted(pairs, key=lambda x: (x["family"], x["chip_id"]))
    return pairs

# ======================================================
# === Step 2: Memory-safe load into 3D array ==========
# ======================================================
def load_sram_log_memsafe(fin_path, rows=2048, columns=2048, chunk_size=2**16, n_cycles=101):
    maping = np.zeros((n_cycles, rows, columns), dtype=np.uint8)
    i = 0

    with open(fin_path, 'r') as fh_in:
        for line in fh_in:
            currentLine = line.strip()
            if not currentLine or any(currentLine.startswith(skip) for skip in [
                "=", "Initializing", "Please", "Options:", "Configuring",
                "Performing", "Enter", "END", "Invalid"
            ]) or re.match(r"^[0-9]\.", currentLine):
                continue

            elif currentLine.startswith("DATA DUMP Cycle:"):
                i = int(re.findall(r'[0-9]+', currentLine)[0])

            else:
                bv = np.empty(rows * columns, dtype=np.uint8)
                pos = 0
                for l in range(0, len(currentLine), chunk_size):
                    part = currentLine[l:l+chunk_size]
                    bits = np.unpackbits(np.frombuffer(bytes.fromhex(part), dtype=np.uint8))
                    end_pos = pos + bits.size
                    if end_pos > bv.size:
                        end_pos = bv.size
                        bits = bits[:end_pos - pos]
                    bv[pos:end_pos] = bits
                    pos = end_pos

                
                maping[i] = bv.reshape(rows, columns)
            
    print(maping.shape)
    return maping


# =========================================
# Convert ONE file
# =========================================
def convert_single(pair, out_root, overwrite=False):
    log_path = pair["log_path"]
    chip_id = pair["chip_id"]
    family = pair["family"]

    # Output path: preserve structure
    out_dir = os.path.join(out_root, family)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{chip_id}.npy")

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {out_path}")
        return

    try:
        print(f"[LOAD] {log_path}")
        data = load_sram_log_memsafe(log_path)  # (101, 2048, 2048)

        # Save as uint8 (compact)
        np.save(out_path, data)

        print(f"[SAVED] {out_path}")

    except Exception as e:
        print(f"[ERROR] {log_path}: {e}")


# =========================================
# Parallel driver
# =========================================
def convert_all(root, out_root, num_workers=None):
    pairs = discover_sram_dataset(root)

    print(f"Total files: {len(pairs)}")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    print(f"Using {num_workers} workers")

    with Pool(num_workers) as pool:
        pool.map(partial(convert_single, out_root=out_root), pairs)


# =========================================
# Main
# =========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Path to log files")
    parser.add_argument("--output_root", required=True, help="Path to save .npy files")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    convert_all(
        root=args.input_root,
        out_root=args.output_root,
        num_workers=args.workers,
    )
