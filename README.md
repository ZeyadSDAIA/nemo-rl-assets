# NeMo-RL Assets Repository

This repository contains essential supplementary files and a **one-click setup script** required for running custom Supervised Fine-Tuning (SFT) using the [NeMo-RL](https://github.com/NVIDIA/nemo-rl) framework.

It is meant to be used alongside the main NeMo-RL project to ensure a **reproducible**, **automated**, and **consistent** setup.

---

## ⚙️ One-Click Setup

Use the included setup script to clone, configure, and initialize NeMo-RL with all assets:

```bash
bash setup_nemo_rl.bash
```

## 📂 Directory Structure

```
.
├── assets/
│   ├── run_sft.bash           # Bash launcher for NeMo-RL SFT
│   ├── run_sft.py             # Custom training entry point
│   ├── sft.yaml               # SFT config
│   └── split_dataset.py       # Utility for train/val dataset splitting
├── setup_nemo_rl.bash         # Full automation script
└── README.md
```

---

## 📄 File Descriptions

### `assets/run_sft.bash`
> **Type**: Shell script  
> **Purpose**: Launches the `run_sft.py` script with custom configurations, dataset paths, and training parameters. Designed to be easily modifiable for various experiments.

### `assets/run_sft.py`
> **Type**: Python script  
> **Purpose**: Updated official `run_sft.py` script to enable loading chat templates via directories.

### `assets/split_dataset.py`
> **Type**: Python script  
> **Purpose**: Splits a full dataset into training and validation sets in `.json` or `.jsonl` format.
