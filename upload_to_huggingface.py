#!/usr/bin/env python3
"""
Upload preprocessed dataset to HuggingFace Hub.

Usage:
    python upload_to_huggingface.py \
        --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
        --repo_id username/dataset-name \
        --private
"""

import argparse
from pathlib import Path
import sys

try:
    from datasets import load_from_disk
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install datasets huggingface_hub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload preprocessed dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload to your personal account (public)
  python upload_to_huggingface.py \\
      --dataset_path preprocessed_data/qwen2_5vl-3b-hope \\
      --repo_id your-username/robospatial2-qwen25vl-3b-tokenized

  # Upload to an organization (private)
  python upload_to_huggingface.py \\
      --dataset_path preprocessed_data/qwen2_5vl-3b-hope \\
      --repo_id your-org/robospatial2-qwen25vl-3b-tokenized \\
      --private

  # Specify token directly
  python upload_to_huggingface.py \\
      --dataset_path preprocessed_data/qwen2_5vl-3b-hope \\
      --repo_id username/dataset-name \\
      --token hf_xxxxxxxxxxxxx
        """
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the preprocessed dataset directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/dataset-name or org/dataset-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private (default: public)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (if not provided, will use cached login)"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload preprocessed tokenized dataset",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch to upload to (default: main)"
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Maximum shard size (default: 500MB)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HuggingFace Dataset Upload")
    print("=" * 80)

    # Check if dataset path exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    print(f"\n📁 Dataset path: {args.dataset_path}")
    print(f"🔗 Repository: {args.repo_id}")
    print(f"🔒 Private: {args.private}")
    print(f"📦 Max shard size: {args.max_shard_size}")

    # Login to HuggingFace
    print("\n[1/4] Authenticating with HuggingFace...")
    try:
        if args.token:
            login(token=args.token)
            print("  ✓ Logged in with provided token")
        else:
            # Try to use cached token
            from huggingface_hub import whoami
            try:
                user_info = whoami()
                print(f"  ✓ Using cached credentials (logged in as: {user_info['name']})")
            except Exception:
                print("  ⚠ No cached credentials found. Please login:")
                login()
    except Exception as e:
        print(f"  ❌ Authentication failed: {e}")
        print("\n  To get your token:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Create a new token with 'write' access")
        print("  3. Run: huggingface-cli login")
        print("  Or provide token with --token argument")
        sys.exit(1)

    # Load dataset
    print(f"\n[2/4] Loading dataset from disk...")
    try:
        dataset = load_from_disk(args.dataset_path)
        print(f"  ✓ Dataset loaded successfully")
        print(f"  Dataset info:")
        if hasattr(dataset, 'keys'):
            for split_name in dataset.keys():
                print(f"    - {split_name}: {len(dataset[split_name])} samples")
        else:
            print(f"    - Total samples: {len(dataset)}")
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        sys.exit(1)

    # Create repository if it doesn't exist
    print(f"\n[3/4] Creating repository (if needed)...")
    try:
        api = HfApi()
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True
        )
        print(f"  ✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"  ⚠ Warning: {e}")
        print("  Continuing with upload...")

    # Upload dataset
    print(f"\n[4/4] Uploading dataset to HuggingFace Hub...")
    print(f"  This may take a while depending on dataset size...")
    try:
        dataset.push_to_hub(
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
            max_shard_size=args.max_shard_size,
        )
        print(f"  ✓ Upload complete!")
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        sys.exit(1)

    # Print success message
    print("\n" + "=" * 80)
    print("✅ Dataset successfully uploaded to HuggingFace Hub!")
    print("=" * 80)
    print(f"\n🔗 Dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    print(f"\n📖 To use this dataset in training:")
    print(f"   1. Load it in Python:")
    print(f"      from datasets import load_dataset")
    print(f"      dataset = load_dataset('{args.repo_id}')")
    print(f"\n   2. Or download to use as tokenized_path:")
    print(f"      from datasets import load_dataset")
    print(f"      dataset = load_dataset('{args.repo_id}')")
    print(f"      dataset.save_to_disk('local_path')")
    print(f"      # Then use: tokenized_path: local_path")

    if args.private:
        print(f"\n🔒 Note: This is a PRIVATE dataset. You'll need to:")
        print(f"   - Be logged in to HuggingFace to access it")
        print(f"   - Use a token with read access")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
