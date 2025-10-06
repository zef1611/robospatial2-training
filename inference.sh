#!/bin/bash

# RoboSpatial2 Inference Script
# Interactive chat interface for the trained model

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}RoboSpatial2 Inference (Chat Mode)${NC}"
echo -e "${GREEN}============================================================${NC}"

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/train_robospatial2.yaml"

# Stay in training directory
cd "$SCRIPT_DIR"

echo -e "\n${YELLOW}Loading model for interactive chat...${NC}"
echo -e "${YELLOW}You can ask questions about camera poses, object positions, etc.${NC}\n"

# Run chat interface
llamafactory-cli chat "$CONFIG_FILE"