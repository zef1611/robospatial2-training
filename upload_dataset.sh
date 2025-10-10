#!/bin/bash

# Upload preprocessed dataset to HuggingFace Hub
# This script makes it easy to upload your tokenized dataset

set -e  # Exit on error

export DISABLE_VERSION_CHECK=1
module load httpproxy python

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Upload Dataset to HuggingFace Hub${NC}"
echo -e "${GREEN}============================================================${NC}"

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
DATASET_PATH="${1:-$SCRIPT_DIR/preprocessed_data/qwen2_5vl-3b-hope}"
REPO_ID="${2}"
PRIVATE="${3:-false}"

# Function to show usage
show_usage() {
    echo -e "\n${YELLOW}Usage:${NC}"
    echo "  $0 <dataset_path> <repo_id> [private]"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Upload public dataset"
    echo "  $0 preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-qwen25vl-3b"
    echo ""
    echo "  # Upload private dataset"
    echo "  $0 preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-qwen25vl-3b true"
    echo ""
    echo "  # Upload to organization"
    echo "  $0 preprocessed_data/qwen2_5vl-3b-hope myorg/robospatial2-qwen25vl-3b"
    echo ""
}

# Check if repo_id is provided
if [ -z "$REPO_ID" ]; then
    echo -e "${RED}Error: Repository ID is required${NC}"
    show_usage
    exit 1
fi

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset path not found: $DATASET_PATH${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Dataset path: $DATASET_PATH"
echo "  Repository ID: $REPO_ID"
echo "  Private: $PRIVATE"

# Check if logged in to HuggingFace
echo -e "\n${BLUE}Checking HuggingFace authentication...${NC}"
if ! python -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
    echo -e "${YELLOW}Not logged in to HuggingFace.${NC}"
    echo -e "${YELLOW}Please login with your HuggingFace token:${NC}"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    huggingface-cli login
fi

# Confirm upload
echo -e "\n${YELLOW}Ready to upload dataset to HuggingFace Hub${NC}"
echo -e "${YELLOW}This will upload the preprocessed dataset to: ${BLUE}$REPO_ID${NC}"
if [ "$PRIVATE" = "true" ]; then
    echo -e "${YELLOW}The dataset will be ${RED}PRIVATE${NC}"
else
    echo -e "${YELLOW}The dataset will be ${GREEN}PUBLIC${NC}"
fi
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Run upload
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Starting upload...${NC}"
echo -e "${YELLOW}This may take a while depending on dataset size...${NC}\n"

if [ "$PRIVATE" = "true" ]; then
    python upload_to_huggingface.py \
        --dataset_path "$DATASET_PATH" \
        --repo_id "$REPO_ID" \
        --private
else
    python upload_to_huggingface.py \
        --dataset_path "$DATASET_PATH" \
        --repo_id "$REPO_ID"
fi

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}✅ Upload completed successfully!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo -e "\n${BLUE}View your dataset at:${NC}"
    echo -e "${GREEN}https://huggingface.co/datasets/$REPO_ID${NC}"
else
    echo -e "\n${RED}Upload failed!${NC}"
    exit 1
fi
