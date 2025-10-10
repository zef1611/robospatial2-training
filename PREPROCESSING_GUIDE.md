# Data Preprocessing Guide

This guide explains how to pre-process and tokenize your dataset to speed up training startup times.

## Problem

When training with LLaMA-Factory, the data preprocessing (tokenization) happens at the start of each training run. For large datasets like HOPE, this can take significant time and needs to be repeated every time you start a new training run.

## Solution

Pre-process and tokenize the dataset **once**, then save it to disk. Subsequent training runs can load the pre-tokenized data directly, skipping the preprocessing step entirely.

## Quick Start

### 1. Run the preprocessing script

```bash
cd robospatial2-training
./preprocess.sh
```

This will:
- Use the default config: `configs/yaml/Qwen2_5-VL-3B-Instruct.yaml`
- Save preprocessed data to: `preprocessed_data/qwen2_5vl-3b-hope/`

### 2. Update your training config

Add this line to your YAML config file:

```yaml
### dataset
tokenized_path: /project/6100855/huyle/projects/robospatial2/robospatial2-training/preprocessed_data/qwen2_5vl-3b-hope
```

Or specify it when running training:

```bash
llamafactory-cli train configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --tokenized_path preprocessed_data/qwen2_5vl-3b-hope
```

### 3. Run training

```bash
./train.sh
```

Training will now start immediately using the pre-tokenized data!

## Advanced Usage

### Custom config and output directory

```bash
./preprocess.sh configs/yaml/MyConfig.yaml preprocessed_data/my-custom-output
```

### Using the Python script directly

```bash
python preprocess_data.py \
    --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --output preprocessed_data/qwen2_5vl-3b-hope
```

### Preprocessing specific datasets

```bash
python preprocess_data.py \
    --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --output preprocessed_data/custom \
    --dataset hope_train,hope_val
```

### Preprocessing only training data (no eval)

```bash
python preprocess_data.py \
    --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --output preprocessed_data/train_only \
    --dataset hope_train
```

## Storage Considerations

Pre-tokenized datasets require disk space:
- **Original HOPE dataset**: ~75 MB (hope_train.json + hope_val.json)
- **Tokenized dataset**: Variable size depending on:
  - Number of samples
  - Sequence length (cutoff_len)
  - Model tokenizer
  - Image/video data references

Estimate: For vision-language models with images, expect 2-5x the original JSON size.

## When to Re-preprocess

You need to re-run preprocessing if you change:

1. **Model tokenizer** - Different models have different tokenizers
2. **Template** - Changing the prompt template changes tokenization
3. **Cutoff length** - Different max sequence lengths require re-tokenization
4. **Dataset** - Adding/modifying training data
5. **Preprocessing parameters** - `preprocessing_num_workers`, data filtering, etc.

## Benefits

✅ **Faster startup**: Skip tokenization on every training run
✅ **Consistency**: Same tokenized data across multiple training runs
✅ **Debugging**: Easier to debug data issues when preprocessing is separate
✅ **Multiple experiments**: Preprocess once, use for multiple training runs with different hyperparameters

## Configuration Options

The preprocessing script respects all dataset-related parameters from your config:

- `dataset` - Training dataset(s)
- `eval_dataset` - Evaluation dataset(s)
- `dataset_dir` - Directory containing dataset files
- `media_dir` - Directory containing images/videos
- `template` - Prompt template to use
- `cutoff_len` - Maximum sequence length
- `preprocessing_num_workers` - Number of parallel workers
- `max_samples` - Limit number of samples (for testing)

## Troubleshooting

### "Output directory already exists"

The script will ask if you want to overwrite. Say 'y' to proceed or 'n' to abort.

### "No module named 'llamafactory'"

Make sure you're in the correct environment and LLaMA-Factory is installed:

```bash
pip install -e LLaMA-Factory/
```

### Preprocessing is still slow

- Increase `preprocessing_num_workers` in your config
- Use a faster storage location (e.g., local SSD instead of network storage)
- Consider using fewer samples for testing first with `max_samples`

### Training fails with preprocessed data

If you change model or tokenizer settings, you must re-preprocess the data. The tokenization must match the model you're training with.

## Example Workflow

```bash
# 1. First time: Preprocess the data
./preprocess.sh

# 2. Add tokenized_path to your config
echo "tokenized_path: preprocessed_data/qwen2_5vl-3b-hope" >> configs/yaml/Qwen2_5-VL-3B-Instruct.yaml

# 3. Run training (fast startup!)
./train.sh

# 4. Run more experiments with same data (no re-preprocessing needed)
# Just modify training hyperparameters like learning_rate, batch_size, etc.
```

## Notes

- The preprocessed dataset is saved using HuggingFace's `datasets.save_to_disk()` format
- You can inspect the preprocessed data using:
  ```python
  from datasets import load_from_disk
  dataset = load_from_disk("preprocessed_data/qwen2_5vl-3b-hope")
  print(dataset)
  ```
- When using `tokenized_path`, most dataset-related arguments are ignored (dataset, eval_dataset, dataset_dir, etc.)
