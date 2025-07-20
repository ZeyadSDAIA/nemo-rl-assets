#!/bin/bash

# ========== 1. Clone NeMo-RL ==========
echo "🚀 Cloning NeMo-RL..."
git clone https://github.com/NVIDIA/nemo-rl.git
cd nemo-rl

# ========== 2. Install uv & Setup venv ==========
echo "🔧 Installing uv and setting up virtual environment..."
pip install uv
uv venv

# ========== 3. Trigger Dependency Installation ==========
echo "📦 Installing NeMo-RL dependencies via script bootstrap..."
uv run python examples/run_sft.py
echo "🔧 Initial run expected to fail due to model authorization. Dependencies are now installed."

# ========== 4. Clone Assets ==========
echo "📂 Cloning NeMo-RL assets..."
git clone https://github.com/ZeyadSDAIA/nemo-rl-assets.git
cp nemo-rl-assets/assets/run_sft.py examples/
cp nemo-rl-assets/assets/sft.yaml examples/configs/
cp nemo-rl-assets/assets/split_dataset.py .
cp nemo-rl-assets/assets/run_sft.bash .

# ========== 5. Delete Assets Repo ==========
echo "🗑️  Cleaning up assets repository..."
rm -rf nemo-rl-assets

# ========== 6. Success Message ==========
echo "🎉 NeMo-RL setup complete! You can now run the SFT script with: "bash run_sft.bash" "
echo "👋 Goodbye! Deleting self..."
cd ..
rm -- test_bash.bash

