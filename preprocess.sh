#!/bin/bash

# RoboSpatial2 Data Preprocessing Script
# This script pre-tokenizes the dataset to speed up training startup

set -e  # Exit on error

export DISABLE_VERSION_CHECK=1
module load httpproxy python gcc arrow cuda/12.6

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}RoboSpatial2 Data Preprocessing${NC}"
echo -e "${GREEN}============================================================${NC}"

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/configs/yaml/Qwen2_5-VL-3B-Instruct.yaml}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/preprocessed_data/qwen2_5vl-3b-hope}"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Config: $CONFIG_FILE"
echo "  Output: $OUTPUT_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "\n${RED}Error: Config file not found at $CONFIG_FILE${NC}"
    echo "Usage: $0 [config_file] [output_dir]"
    exit 1
fi

# Check if data exists
DATA_DIR="$SCRIPT_DIR/data"
if [ ! -f "$DATA_DIR/hope_train.json" ]; then
    echo -e "\n${YELLOW}Warning: hope_train.json not found. Running conversion script...${NC}"
    python "$SCRIPT_DIR/convert_dataset.py"
fi

# Run preprocessing
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Starting preprocessing...${NC}"
echo -e "${YELLOW}This will tokenize the dataset and save it for faster training startup${NC}\n"

python preprocess_data.py \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}Preprocessing completed successfully!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "1. Update your training config to use the preprocessed data:"
    echo "   Add this line to $CONFIG_FILE:"
    echo -e "   ${GREEN}tokenized_path: $OUTPUT_DIR${NC}"
    echo ""
    echo "2. Or run training with the tokenized data:"
    echo -e "   ${GREEN}llamafactory-cli train $CONFIG_FILE --tokenized_path $OUTPUT_DIR${NC}"
else
    echo -e "\n${RED}Preprocessing failed!${NC}"
    exit 1
fi
