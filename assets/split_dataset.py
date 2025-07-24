import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

input_path = Path(sys.argv[1]).resolve()
train_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

# Load data
ext = input_path.suffix.lower()
with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f] if ext == ".jsonl" else json.load(f)

# Simple random split - no filtering
train_data, val_data = train_test_split(data, test_size=(1.0 - train_ratio), random_state=42)

# Save
output_dir = input_path.parent / "tmp"
output_dir.mkdir(parents=True, exist_ok=True)

train_path = output_dir / "train_split.jsonl"
val_path = output_dir / "val_split.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(train_path)
print(val_path)