#!/bin/bash

# RoboSpatial2 Distributed Training Script (Multi-Node)
# This script trains the Qwen2.5-VL model on the HOPE dataset across multiple nodes
export DISABLE_VERSION_CHECK=1
module load httpproxy python gcc arrow cuda/12.6
set -e  # Exit on error

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

echo -e "${YELLOW}Running distributed training on Node $NODE_RANK...${NC}"
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train_distributed_standalone_test.py