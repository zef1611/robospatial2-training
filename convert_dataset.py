#!/usr/bin/env python3
"""
Convert HOPE dataset from JSONL to training JSON format.
Converts the conversations format to messages format with proper role tags.
"""
import json
import os
from pathlib import Path

def convert_jsonl_to_json(input_file, output_file, prefix):
    """Convert JSONL format to JSON array format for training."""
    data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)

                # Convert "from" to "role" format
                messages = []
                for conv in item.get('conversations', []):
                    messages.append({
                        'role': 'user' if conv['from'] == 'user' else 'assistant',
                        'content': conv['value']
                    })

                # Create new format
                new_item = {
                    'messages': messages,
                    'images': item.get('images', [])
                }

                full_paths = [os.path.join(prefix, f) for f in new_item['images']]
                new_item['images'] = full_paths            

                # Add optional metadata
                if 'question_type' in item:
                    new_item['question_type'] = item['question_type']
                if 'question_subtype' in item:
                    new_item['question_subtype'] = item['question_subtype']

                data.append(new_item)

    # Write as JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(data)} samples from {input_file} to {output_file}")
    return len(data)

def main():
    # Setup paths
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    training_dir = Path(__file__).parent
    data_dir = training_dir / "data"

    # Ensure output directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Converting HOPE datasets to training format...")
    print("=" * 60)

    # Convert training dataset
    train_input = dataset_dir / "hope-train-qa" / "train.jsonl"
    train_output = data_dir / "hope_train.json"
    train_count = convert_jsonl_to_json(train_input, train_output, 'hope-train-qa')

    # Convert validation dataset
    val_input = dataset_dir / "hope-val-qa" / "val.jsonl"
    val_output = data_dir / "hope_val.json"
    val_count = convert_jsonl_to_json(val_input, val_output, 'hope-val-qa')

    print("=" * 60)
    print(f"Conversion complete!")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"\nOutput files:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print("\nNext steps:")
    print("  1. Update robospatial2-training/data/dataset_info.json with the new datasets")
    print("  2. Run the training script")

if __name__ == "__main__":
    main()