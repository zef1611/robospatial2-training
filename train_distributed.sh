#!/bin/bash

# RoboSpatial2 Distributed Training Script (Multi-Node)
# This script trains the Qwen2.5-VL model on the HOPE dataset across multiple nodes
export DISABLE_VERSION_CHECK=1
module load httpproxy python gcc arrow cuda/12.6
set -e  # Exit on error

export WANDB_API_KEY=8a61187e67668d582555b5b1dca57f666030bb15
export WANDB_PROJECT=rs2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}RoboSpatial2 Distributed Training Script (Multi-Node)${NC}"
echo -e "${GREEN}============================================================${NC}"

# Distributed training configuration
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-2}
MASTER_ADDR=${MASTER_ADDR:-"tg10602"}
MASTER_PORT=${MASTER_PORT:-29500}

echo -e "\n${YELLOW}Distributed Configuration:${NC}"
echo "  Total Nodes: $NNODES"
echo "  Current Node Rank: $NODE_RANK"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"

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

# Check if datasets exist (only on master node)
if [ $NODE_RANK -eq 0 ]; then
    DATA_DIR="$SCRIPT_DIR/data"
    if [ ! -f "$DATA_DIR/hope_train.json" ]; then
        echo -e "\n${YELLOW}Warning: hope_train.json not found. Running conversion script...${NC}"
        python "$SCRIPT_DIR/convert_dataset.py"
    fi
fi

# Stay in training directory
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Starting distributed training...${NC}"
echo -e "${YELLOW}Note: This will use DeepSpeed ZeRO-3 optimization across multiple nodes${NC}\n"

# Check number of GPUs on this node
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo -e "${RED}Error: No GPUs detected. This training requires GPU(s).${NC}"
    exit 1
fi

echo -e "${GREEN}Node $NODE_RANK: Detected $NUM_GPUS GPU(s)${NC}\n"

# Check if wandb is installed (only on master node)
if [ $NODE_RANK -eq 0 ]; then
    if ! python -c "import wandb" 2>/dev/null; then
        echo -e "${YELLOW}Warning: wandb not installed. Installing...${NC}"
        pip install wandb
    fi

    # Check if wandb is logged in
    if ! wandb login --relogin 2>/dev/null | grep -q "Successfully logged in"; then
        echo -e "${YELLOW}Please login to Weights & Biases:${NC}"
        wandb login
    fi
fi

# Run distributed training with llamafactory-cli
echo -e "${YELLOW}Running distributed training on Node $NODE_RANK...${NC}"
FORCE_TORCHRUN=1 NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    llamafactory-cli train "$CONFIG_FILE"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Training completed on Node $NODE_RANK!${NC}"
echo -e "${GREEN}============================================================${NC}"

if [ $NODE_RANK -eq 0 ]; then
    echo -e "\nModel saved to: $SCRIPT_DIR/saves/robospatial2-qwen2_5vl-7b/full/sft"
    echo -e "\nTo evaluate the model, use the evaluation script or run:"
    echo -e "  cd $SCRIPT_DIR"
    echo -e "  llamafactory-cli chat $CONFIG_FILE"
fi