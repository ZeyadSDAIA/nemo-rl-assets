#!/usr/bin/env python3
"""
High-performance JSONL splitter using local SSD.
- Reads from external SSD
- Splits and writes to fast local storage
- Does NOT copy back
- Outputs ONLY train/val paths to stdout (for bash capture)

Usage: split_dataset.py <input.jsonl> <train_ratio> <seed> <tmp_root_dir>
"""

import sys
import random
import time
from pathlib import Path
from tqdm import tqdm


# Buffer size for high-performance writes
BUFFER_SIZE = 32 * 1024 * 1024  # 32MB


def main():
    # === Parse arguments ===
    if len(sys.argv) != 5:
        print("❌ Usage: split_dataset.py <input.jsonl> <train_ratio> <seed> <tmp_root_dir>", file=sys.stderr)
        sys.exit(1)

    try:
        input_path = Path(sys.argv[1]).resolve()
        train_ratio = float(sys.argv[2])
        seed = int(sys.argv[3])
        tmp_root_dir = Path(sys.argv[4]).resolve()
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}", file=sys.stderr)
        sys.exit(1)

    # === Validate inputs ===
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not (0.0 < train_ratio < 1.0):
        print("❌ Error: train_ratio must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    # === Setup unique working directory on fast local disk ===
    try:
        tmp_root_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root_dir / f"split_{input_path.stem}_{int(time.time())}_{random.randint(10000, 99999)}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
    except Exception as e:
        print(f"❌ Failed to create temp directory: {e}", file=sys.stderr)
        sys.exit(1)

    train_file_local = tmp_dir / "train_split.jsonl"
    val_file_local = tmp_dir / "val_split.jsonl"

    try:
        # === Phase 1: Load from external SSD ===
        t0 = time.time()
        print("📦 Loading dataset from external SSD...", file=sys.stderr, flush=True)
        try:
            data = input_path.read_bytes().splitlines(keepends=True)
            lines = [line for line in data if line.strip()]
            total = len(lines)
            load_time = time.time() - t0
            print(f"✅ Loaded {total:,} lines in {load_time:.2f}s", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to load input file: {e}", file=sys.stderr)
            raise

        # === Phase 2: Shuffle and split ===
        t1 = time.time()
        print("🔀 Shuffling and splitting...", file=sys.stderr, flush=True)
        try:
            random.seed(seed)
            random.shuffle(lines)
            split_idx = int(total * train_ratio)
            train_lines = lines[:split_idx]
            val_lines = lines[split_idx:]
            split_time = time.time() - t1
            print(f"✅ Split: {len(train_lines):,} train | {len(val_lines):,} val in {split_time:.2f}s",
                  file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to shuffle or split: {e}", file=sys.stderr)
            raise

        # === Phase 3: Write splits to LOCAL disk (fast) ===
        t2 = time.time()
        print("💾 Writing splits to local storage...", file=sys.stderr, flush=True)

        # Write train
        try:
            with open(train_file_local, 'wb', buffering=BUFFER_SIZE) as f:
                for line in tqdm(train_lines, desc="📝 Writing train", unit="lines", ncols=80, file=sys.stderr):
                    f.write(line)
                f.flush()  # Ensure all data is written
                import os
                os.fsync(f.fileno())  # Force write to disk
        except Exception as e:
            print(f"❌ Failed to write train file: {e}", file=sys.stderr)
            raise

        # Write val
        try:
            with open(val_file_local, 'wb', buffering=BUFFER_SIZE) as f:
                for line in tqdm(val_lines, desc="📝 Writing val  ", unit="lines", ncols=80, file=sys.stderr):
                    f.write(line)
                f.flush()
                import os
                os.fsync(f.fileno())
        except Exception as e:
            print(f"❌ Failed to write val file: {e}", file=sys.stderr)
            raise

        write_time = time.time() - t2
        print(f"✅ Wrote splits to {tmp_dir} in {write_time:.2f}s", file=sys.stderr)

        # === FINAL OUTPUT: ONLY THESE TWO LINES TO stdout ===
        # They must be last, clean, and not mixed with logs
        print(train_file_local)
        print(val_file_local)
        # ====================================================

        # Signal cleanup info to stderr
        print(f"📌 SPLIT_DIR={tmp_dir}", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"❌ Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()