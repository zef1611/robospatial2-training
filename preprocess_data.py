#!/usr/bin/env python3
"""
Pre-process and tokenize data for LLaMA-Factory training.
This script tokenizes the dataset once and saves it to disk,
so you don't need to re-tokenize every time you start training.

Usage:
    # With GPU:
    python preprocess_data.py --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml --output preprocessed_data/qwen2_5vl-3b

    # CPU-only (for CPU clusters):
    python preprocess_data.py --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml --output preprocessed_data/qwen2_5vl-3b --cpu
"""

import argparse
import os
import sys
from pathlib import Path

# Set environment variables to simulate single-process distributed setup
# This bypasses the distributed training check in LLaMA-Factory
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Add LLaMA-Factory to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LLAMAFACTORY_PATH = PROJECT_ROOT / "LLaMA-Factory" / "src"
sys.path.insert(0, str(LLAMAFACTORY_PATH))

from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Pre-process and tokenize training data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save the tokenized dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to preprocess (overrides config, comma-separated for multiple datasets)"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Eval dataset name to preprocess (overrides config, comma-separated for multiple datasets)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only execution (for CPU clusters without GPU)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LLaMA-Factory Data Preprocessing Script")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.output}")
    if args.cpu:
        print(f"Mode: CPU-only")

    # Check if output directory already exists
    output_path = Path(args.output)
    if output_path.exists():
        response = input(f"\nOutput directory '{args.output}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Load training arguments from config
    print("\n[1/5] Loading configuration...")
    sys.argv = ["preprocess_data.py", args.config]

    # Add tokenized_path to save the preprocessed data
    sys.argv.extend(["tokenized_path=" + args.output])

    # Override dataset if provided
    if args.dataset:
        sys.argv.append(f"dataset={args.dataset}")
    if args.eval_dataset:
        sys.argv.append(f"eval_dataset={args.eval_dataset}")

    # Disable training, enable saving, and disable DeepSpeed for preprocessing
    sys.argv.extend([
        "do_train=false",
        "output_dir=/tmp/llamafactory_preprocess_temp",  # Dummy output dir
        "deepspeed=null",  # Disable DeepSpeed for preprocessing
    ])

    # Add CPU-specific settings if requested
    if args.cpu:
        sys.argv.extend([
            "no_cuda=true",      # Force CPU usage
            "fp16=false",        # Disable FP16 (not supported on CPU)
            "bf16=false",        # Disable BF16 (not supported on CPU)
        ])

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  Stage: {finetuning_args.stage}")
    print(f"  Dataset: {data_args.dataset}")
    if data_args.eval_dataset:
        print(f"  Eval Dataset: {data_args.eval_dataset}")

    # Load tokenizer and model processor
    print("\n[2/5] Loading tokenizer and processor...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    # Get template
    print("\n[3/5] Loading template...")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load and preprocess dataset
    print("\n[4/5] Loading and tokenizing dataset...")
    print("  This may take a while depending on dataset size...")
    print(f"  Using {data_args.preprocessing_num_workers} workers for preprocessing")

    # Set do_train to True to trigger dataset preprocessing
    training_args.do_train = True

    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage=finetuning_args.stage,
        tokenizer=tokenizer,
        processor=processor,
    )

    # Double-check that the dataset was saved
    # If get_dataset didn't save it, we'll save it manually
    if not output_path.exists() or not any(output_path.iterdir()):
        print("  Saving dataset manually...")
        from datasets import DatasetDict

        # Create dataset dict from the module
        dataset_dict = DatasetDict()
        if hasattr(dataset_module, "train_dataset") and dataset_module.train_dataset is not None:
            dataset_dict["train"] = dataset_module.train_dataset
        if hasattr(dataset_module, "eval_dataset") and dataset_module.eval_dataset is not None:
            dataset_dict["eval"] = dataset_module.eval_dataset

        # Save to disk
        dataset_dict.save_to_disk(args.output)
        print(f"  Manually saved dataset to {args.output}")

    print("\n[5/5] Dataset preprocessing complete!")
    print("=" * 80)
    print(f"✓ Tokenized dataset saved to: {args.output}")
    print("\nTo use the preprocessed dataset in training, add this to your config:")
    print(f"  tokenized_path: {args.output}")
    print("\nOr use it from command line:")
    print(f"  llamafactory-cli train config.yaml --tokenized_path {args.output}")
    print("=" * 80)

    # Print dataset statistics
    if hasattr(dataset_module, "train_dataset") and dataset_module.train_dataset is not None:
        print(f"\nTrain samples: {len(dataset_module.train_dataset)}")
    if hasattr(dataset_module, "eval_dataset") and dataset_module.eval_dataset is not None:
        print(f"Eval samples: {len(dataset_module.eval_dataset)}")


if __name__ == "__main__":
    main()
