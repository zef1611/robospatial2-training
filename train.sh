#!/bin/bash

# RoboSpatial2 Training Script
# This script trains the Qwen2.5-VL model on the HOPE dataset for robotic spatial understanding
export DISABLE_VERSION_CHECK=1
module load httpproxy python gcc arrow cuda/12.6
set -e  # Exit on error

export WANDB_API_KEY=
export WANDB_PROJECT=
python -V
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}RoboSpatial2 Training Script${NC}"
echo -e "${GREEN}============================================================${NC}"

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/configs/yaml/Qwen2_5-VL-3B-Instruct.yaml"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Project Root: $PROJECT_ROOT"
echo "  Training Dir: $SCRIPT_DIR"
echo "  Config File: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "\n${RED}Error: Config file not found at $CONFIG_FILE${NC}"
    exit 1
fi

# Check if datasets exist
DATA_DIR="$SCRIPT_DIR/data"
if [ ! -f "$DATA_DIR/hope_train.json" ]; then
    echo -e "\n${YELLOW}Warning: hope_train.json not found. Running conversion script...${NC}"
    python "$SCRIPT_DIR/convert_dataset.py"
fi

# Stay in training directory
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Starting training...${NC}"
echo -e "${YELLOW}Note: This will use DeepSpeed ZeRO-3 optimization${NC}\n"

# Check number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo -e "${RED}Error: No GPUs detected. This training requires GPU(s).${NC}"
    exit 1
fi

echo -e "${GREEN}Detected $NUM_GPUS GPU(s)${NC}\n"

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo -e "${YELLOW}Warning: wandb not installed. Installing...${NC}"
    pip install wandb
fi

# Check if wandb is logged in
if ! wandb login --relogin 2>/dev/null | grep -q "Successfully logged in"; then
    echo -e "${YELLOW}Please login to Weights & Biases:${NC}"
    wandb login
fi

# Run training with llamafactory-cli
if [ $NUM_GPUS -gt 1 ]; then
    echo -e "${YELLOW}Running multi-GPU training with $NUM_GPUS GPUs...${NC}"
    llamafactory-cli train "$CONFIG_FILE"
else
    echo -e "${YELLOW}Running single-GPU training...${NC}"
    llamafactory-cli train "$CONFIG_FILE"
fi

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "\nModel saved to: $SCRIPT_DIR/saves/robospatial2-qwen2_5vl-7b/full/sft"
echo -e "\nTo evaluate the model, use the evaluation script or run:"
echo -e "  cd $SCRIPT_DIR"
echo -e "  llamafactory-cli chat $CONFIG_FILE"