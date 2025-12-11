#!/bin/bash
# Quick setup script for cloud training environments
# Run this after cloning the repo on Vertex AI / Colab

set -e

echo "=============================================="
echo "Memory-Augmented Transformer - Cloud Setup"
echo "=============================================="

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: No GPU detected!"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers datasets accelerate wandb tokenizers einops safetensors pyyaml tqdm scikit-learn

# Optional: Flash Attention (uncomment for speed boost on A100)
# pip install flash-attn --no-build-isolation

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test model creation
echo ""
echo "Testing model creation..."
python -c "
import sys
sys.path.insert(0, '.')
from memory_transformer.model.config import MemoryTransformerConfig
from memory_transformer.model.memory_transformer import MemoryTransformer
config = MemoryTransformerConfig()
model = MemoryTransformer(config)
params = sum(p.numel() for p in model.parameters())
print(f'Model created: {params/1e6:.1f}M parameters')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Quick start options:"
echo ""
echo "1. Quick validation (~1 hour, ~\$4):"
echo "   python scripts/train_cloud.py --model-config configs/tiny_full.yaml --max-steps 5000"
echo ""
echo "2. Medium model (~6 hours, ~\$25):"
echo "   python scripts/train_cloud.py --model-config configs/medium_a100.yaml --max-steps 50000"
echo ""
echo "3. Hyperparameter sweep (parallel):"
echo "   wandb sweep configs/sweep_config.yaml"
echo ""
echo "4. Multi-GPU (8x A100, ~3 hours, ~\$90):"
echo "   accelerate launch --num_processes=8 scripts/train_distributed.py"
echo ""
