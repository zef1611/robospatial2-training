#!/usr/bin/env python3
"""
Download preprocessed dataset from HuggingFace Hub.

Usage:
    python download_from_huggingface.py \
        --repo_id username/dataset-name \
        --output_path preprocessed_data/downloaded
"""

import argparse
from pathlib import Path
import sys

try:
    from datasets import load_dataset
    from huggingface_hub import login
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install datasets huggingface_hub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download preprocessed dataset from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download public dataset
  python download_from_huggingface.py \\
      --repo_id username/robospatial2-qwen25vl-3b-tokenized \\
      --output_path preprocessed_data/qwen2_5vl-3b-hope

  # Download private dataset with token
  python download_from_huggingface.py \\
      --repo_id username/robospatial2-qwen25vl-3b-tokenized \\
      --output_path preprocessed_data/qwen2_5vl-3b-hope \\
      --token hf_xxxxxxxxxxxxx

  # Download specific split
  python download_from_huggingface.py \\
      --repo_id username/dataset-name \\
      --output_path preprocessed_data/train_only \\
      --split train
        """
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/dataset-name)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the dataset will be saved"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (for private datasets)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Specific split to download (e.g., 'train', 'test'). If not specified, downloads all splits."
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HuggingFace Dataset Download")
    print("=" * 80)

    output_path = Path(args.output_path)

    print(f"\n🔗 Repository: {args.repo_id}")
    print(f"📁 Output path: {args.output_path}")
    if args.split:
        print(f"📊 Split: {args.split}")

    # Check if output path already exists
    if output_path.exists():
        response = input(f"\n⚠️  Output path '{args.output_path}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(0)

    # Login if token provided
    if args.token:
        print("\n[1/3] Authenticating with HuggingFace...")
        try:
            login(token=args.token)
            print("  ✓ Logged in with provided token")
        except Exception as e:
            print(f"  ❌ Authentication failed: {e}")
            sys.exit(1)
    else:
        print("\n[1/3] Using cached credentials (if any)...")

    # Download dataset
    print(f"\n[2/3] Downloading dataset from HuggingFace Hub...")
    print(f"  This may take a while depending on dataset size...")
    try:
        dataset = load_dataset(
            args.repo_id,
            split=args.split,
            token=args.token,
        )
        print(f"  ✓ Dataset downloaded successfully")

        if args.split:
            print(f"    - {args.split}: {len(dataset)} samples")
        elif hasattr(dataset, 'keys'):
            print(f"  Dataset splits:")
            for split_name in dataset.keys():
                print(f"    - {split_name}: {len(dataset[split_name])} samples")
        else:
            print(f"    - Total samples: {len(dataset)}")

    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        if "401" in str(e) or "403" in str(e):
            print("\n  This might be a private dataset. Please:")
            print("  1. Get your token from: https://huggingface.co/settings/tokens")
            print("  2. Use: python download_from_huggingface.py --token YOUR_TOKEN ...")
        sys.exit(1)

    # Save to disk
    print(f"\n[3/3] Saving dataset to disk...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(args.output_path)
        print(f"  ✓ Dataset saved to: {args.output_path}")
    except Exception as e:
        print(f"  ❌ Failed to save dataset: {e}")
        sys.exit(1)

    # Print success message
    print("\n" + "=" * 80)
    print("✅ Dataset successfully downloaded!")
    print("=" * 80)
    print(f"\n📖 To use this dataset in training:")
    print(f"   Add to your config file:")
    print(f"   tokenized_path: {args.output_path}")
    print(f"\n   Or use command line:")
    print(f"   llamafactory-cli train config.yaml --tokenized_path {args.output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
