#!/bin/bash

# RoboSpatial2 Evaluation Script
# Evaluates the trained model on the validation set

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}RoboSpatial2 Evaluation Script${NC}"
echo -e "${GREEN}============================================================${NC}"

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/train_robospatial2.yaml"

# Check if checkpoint exists
CHECKPOINT_DIR="$SCRIPT_DIR/saves/robospatial2-qwen2_5vl-7b/full/sft"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${RED}Error: No trained model found at $CHECKPOINT_DIR${NC}"
    echo -e "${YELLOW}Please run training first: ./train.sh${NC}"
    exit 1
fi

# Stay in training directory
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Running evaluation on validation set...${NC}\n"

# Run evaluation
llamafactory-cli eval "$CONFIG_FILE"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Evaluation completed!${NC}"
echo -e "${GREEN}============================================================${NC}"