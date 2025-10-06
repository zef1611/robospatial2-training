#!/usr/bin/env python3
"""
RoboSpatial2 Standalone Distributed Training Script

This script trains the Qwen2.5-VL model on the HOPE dataset using LLaMA-Factory library directly,
without calling the CLI. It supports multi-node distributed training with DeepSpeed ZeRO-3.

Usage:
    # Single node:
    FORCE_TORCHRUN=1 torchrun --nproc_per_node=8 train_distributed_standalone.py

    # Multi-node (on each node):
    NODE_RANK=0 NNODES=2 MASTER_ADDR=tg10602 MASTER_PORT=29500 \
        FORCE_TORCHRUN=1 torchrun --nproc_per_node=8 \
        --nnodes=2 --node_rank=0 --master_addr=tg10602 --master_port=29500 \
        train_distributed_standalone.py
"""

import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Add LLaMA-Factory to path if needed
llamafactory_path = Path(__file__).parent.parent / "LLaMA-Factory" / "src"
if llamafactory_path.exists():
    sys.path.insert(0, str(llamafactory_path))

from llamafactory.train.tuner import run_exp


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    dict_config = OmegaConf.load(config_path)
    return OmegaConf.to_container(dict_config, resolve=True)


def setup_environment():
    """Setup environment variables and modules."""
    # Disable version check
    os.environ["DISABLE_VERSION_CHECK"] = "1"

    # Set distributed training environment variables if not already set
    os.environ.setdefault("NODE_RANK", str(os.getenv("NODE_RANK", "0")))
    os.environ.setdefault("NNODES", str(os.getenv("NNODES", "1")))
    os.environ.setdefault("MASTER_ADDR", os.getenv("MASTER_ADDR", "localhost"))
    os.environ.setdefault("MASTER_PORT", str(os.getenv("MASTER_PORT", "29500")))

    # Force torchrun mode for DeepSpeed
    os.environ["FORCE_TORCHRUN"] = "1"

    return {
        "node_rank": int(os.environ["NODE_RANK"]),
        "nnodes": int(os.environ["NNODES"]),
        "master_addr": os.environ["MASTER_ADDR"],
        "master_port": os.environ["MASTER_PORT"],
    }


def check_data_files(script_dir: Path, node_rank: int):
    """Check if dataset files exist, convert if needed (master node only)."""
    if node_rank == 0:
        data_dir = script_dir / "data"
        if not (data_dir / "hope_train.json").exists():
            print(f"\033[1;33mWarning: hope_train.json not found. Running conversion script...\033[0m")
            import subprocess
            subprocess.run([sys.executable, str(script_dir / "convert_dataset.py")], check=True)


def main():
    """Main training function."""
    # Colors for output
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

    print(f"{GREEN}============================================================{NC}")
    print(f"{GREEN}RoboSpatial2 Standalone Distributed Training Script{NC}")
    print(f"{GREEN}============================================================{NC}")

    # Setup environment
    dist_config = setup_environment()

    print(f"\n{YELLOW}Distributed Configuration:{NC}")
    print(f"  Total Nodes: {dist_config['nnodes']}")
    print(f"  Current Node Rank: {dist_config['node_rank']}")
    print(f"  Master Address: {dist_config['master_addr']}")
    print(f"  Master Port: {dist_config['master_port']}")

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    config_file = script_dir / "configs" / "yaml" / "Qwen2_5-VL-3B-Instruct.yaml"

    print(f"\n{YELLOW}Configuration:{NC}")
    print(f"  Project Root: {project_root}")
    print(f"  Training Dir: {script_dir}")
    print(f"  Config File: {config_file}")

    # Check if config file exists
    if not config_file.exists():
        print(f"\n{RED}Error: Config file not found at {config_file}{NC}")
        sys.exit(1)

    # Check datasets (only on master node)
    try:
        check_data_files(script_dir, dist_config['node_rank'])
    except Exception as e:
        if dist_config['node_rank'] == 0:
            print(f"{YELLOW}Warning: Could not verify/convert dataset: {e}{NC}")

    # Change to training directory
    os.chdir(script_dir)

    print(f"\n{GREEN}Starting distributed training...{NC}")
    print(f"{YELLOW}Note: This will use DeepSpeed ZeRO-3 optimization across multiple nodes{NC}\n")

    # Check number of GPUs
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print(f"{RED}Error: No GPUs detected. This training requires GPU(s).{NC}")
            sys.exit(1)
        print(f"{GREEN}Node {dist_config['node_rank']}: Detected {num_gpus} GPU(s){NC}\n")
    except Exception as e:
        print(f"{RED}Error checking GPUs: {e}{NC}")
        sys.exit(1)

    # Load configuration
    print(f"{YELLOW}Loading configuration from {config_file}...{NC}")
    try:
        config = load_config(str(config_file))
        print(f"{GREEN}Configuration loaded successfully{NC}\n")
    except Exception as e:
        print(f"{RED}Error loading configuration: {e}{NC}")
        sys.exit(1)

    # Setup wandb (only on master node)
    if dist_config['node_rank'] == 0:
        try:
            import wandb
            # Check if wandb API key is set
            if not os.getenv("WANDB_API_KEY"):
                print(f"{YELLOW}Warning: WANDB_API_KEY not set. Wandb logging may not work.{NC}")
            else:
                print(f"{GREEN}Wandb configured{NC}")
        except ImportError:
            print(f"{YELLOW}Warning: wandb not installed. Install with: pip install wandb{NC}")

    # Run training using LLaMA-Factory library
    print(f"{YELLOW}Running distributed training on Node {dist_config['node_rank']}...{NC}\n")

    try:
        # Call run_exp with the loaded configuration
        run_exp(args=config)

        print(f"\n{GREEN}============================================================{NC}")
        print(f"{GREEN}Training completed on Node {dist_config['node_rank']}!{NC}")
        print(f"{GREEN}============================================================{NC}")

        if dist_config['node_rank'] == 0:
            output_dir = config.get('output_dir', script_dir / 'saves' / 'robospatial2-qwen2_5vl-7b' / 'full' / 'sft')
            print(f"\nModel saved to: {output_dir}")
            print(f"\nTo evaluate the model, use the evaluation script or run:")
            print(f"  cd {script_dir}")
            print(f"  llamafactory-cli chat {config_file}")

    except Exception as e:
        print(f"\n{RED}============================================================{NC}")
        print(f"{RED}Training failed on Node {dist_config['node_rank']}!{NC}")
        print(f"{RED}Error: {e}{NC}")
        print(f"{RED}============================================================{NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
