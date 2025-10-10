# Scripts Summary

This document provides a quick reference for all preprocessing and HuggingFace scripts.

## 📋 Available Scripts

### Preprocessing Scripts

#### `preprocess.sh`
**Purpose**: Preprocess and tokenize data for faster training startup

**Usage**:
```bash
# Use default config
./preprocess.sh

# Use custom config and output
./preprocess.sh configs/yaml/MyConfig.yaml preprocessed_data/my-output
```

**What it does**:
- Loads your training config
- Tokenizes the dataset
- Saves to disk for reuse

---

#### `preprocess_data.py`
**Purpose**: Python script for data preprocessing with more control

**Usage**:
```bash
# Basic usage
python preprocess_data.py \
    --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --output preprocessed_data/qwen2_5vl-3b-hope

# Override datasets
python preprocess_data.py \
    --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --output preprocessed_data/custom \
    --dataset hope_train \
    --eval-dataset hope_val
```

**Options**:
- `--config`: Path to training config YAML
- `--output`: Output directory for tokenized data
- `--dataset`: Override dataset name(s)
- `--eval-dataset`: Override eval dataset name(s)

---

### HuggingFace Upload Scripts

#### `upload_dataset.sh`
**Purpose**: Easy upload of preprocessed data to HuggingFace Hub

**Usage**:
```bash
# Upload public dataset
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/dataset-name

# Upload private dataset
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/dataset-name true

# Upload to organization
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope myorg/dataset-name
```

**Arguments**:
1. Dataset path (local preprocessed data directory)
2. Repository ID (username/dataset-name or org/dataset-name)
3. Private flag (optional, default: false)

---

#### `upload_to_huggingface.py`
**Purpose**: Python script for HuggingFace upload with advanced options

**Usage**:
```bash
# Basic upload
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
    --repo_id username/dataset-name

# Upload private dataset
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
    --repo_id username/dataset-name \
    --private

# With custom settings
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
    --repo_id username/dataset-name \
    --commit_message "Updated dataset" \
    --max_shard_size 1GB \
    --token hf_xxxxxxxxxxxxx
```

**Options**:
- `--dataset_path`: Local dataset directory
- `--repo_id`: HuggingFace repository ID
- `--private`: Make dataset private
- `--token`: HuggingFace API token
- `--commit_message`: Custom commit message
- `--max_shard_size`: Maximum file shard size
- `--branch`: Target branch (default: main)

---

### HuggingFace Download Scripts

#### `download_from_huggingface.py`
**Purpose**: Download preprocessed datasets from HuggingFace Hub

**Usage**:
```bash
# Download public dataset
python download_from_huggingface.py \
    --repo_id username/dataset-name \
    --output_path preprocessed_data/downloaded

# Download private dataset
python download_from_huggingface.py \
    --repo_id username/dataset-name \
    --output_path preprocessed_data/downloaded \
    --token hf_xxxxxxxxxxxxx

# Download specific split
python download_from_huggingface.py \
    --repo_id username/dataset-name \
    --output_path preprocessed_data/train_only \
    --split train
```

**Options**:
- `--repo_id`: HuggingFace repository ID
- `--output_path`: Where to save the dataset
- `--token`: HuggingFace API token (for private datasets)
- `--split`: Specific split to download (optional)

---

## 🔄 Common Workflows

### Workflow 1: Local Preprocessing Only

```bash
# Preprocess data
./preprocess.sh

# Update config to use preprocessed data
echo "tokenized_path: preprocessed_data/qwen2_5vl-3b-hope" >> configs/yaml/Qwen2_5-VL-3B-Instruct.yaml

# Train
./train.sh
```

---

### Workflow 2: Preprocess and Upload to HuggingFace

```bash
# 1. Preprocess data
./preprocess.sh

# 2. Login to HuggingFace (one-time)
huggingface-cli login

# 3. Upload to HuggingFace
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-tokenized

# 4. Share the HuggingFace link with your team!
```

---

### Workflow 3: Download and Use Preprocessed Data

```bash
# 1. Download from HuggingFace
python download_from_huggingface.py \
    --repo_id username/robospatial2-tokenized \
    --output_path preprocessed_data/downloaded

# 2. Train with downloaded data
llamafactory-cli train config.yaml \
    --tokenized_path preprocessed_data/downloaded
```

---

### Workflow 4: Multi-Machine Training

**On preprocessing machine:**
```bash
# Preprocess once
./preprocess.sh

# Upload to HuggingFace
./upload_dataset.sh \
    preprocessed_data/qwen2_5vl-3b-hope \
    username/robospatial2-tokenized
```

**On training machines (multiple GPUs):**
```bash
# Download preprocessed data
python download_from_huggingface.py \
    --repo_id username/robospatial2-tokenized \
    --output_path preprocessed_data/tokenized

# Train on each machine
./train.sh  # Uses config with tokenized_path
```

---

## 🎯 Quick Reference Table

| Task | Script | Example |
|------|--------|---------|
| Preprocess data | `./preprocess.sh` | `./preprocess.sh` |
| Preprocess with custom config | `./preprocess.sh` | `./preprocess.sh config.yaml output/` |
| Upload to HuggingFace | `./upload_dataset.sh` | `./upload_dataset.sh data/ user/name` |
| Upload private dataset | `./upload_dataset.sh` | `./upload_dataset.sh data/ user/name true` |
| Download from HuggingFace | `download_from_huggingface.py` | `python download_from_huggingface.py --repo_id user/name --output_path data/` |

---

## 📖 Documentation Files

| File | Content |
|------|---------|
| `README_PREPROCESSING.md` | Quick start guide for preprocessing |
| `PREPROCESSING_GUIDE.md` | Detailed preprocessing documentation |
| `QUICKSTART_HUGGINGFACE.md` | Quick start for HuggingFace upload/download |
| `HUGGINGFACE_GUIDE.md` | Complete HuggingFace documentation |
| `SCRIPTS_SUMMARY.md` | This file - scripts reference |

---

## 🛠️ Prerequisites

### For Preprocessing
```bash
pip install datasets transformers
```

### For HuggingFace Upload/Download
```bash
pip install datasets huggingface_hub

# One-time login
huggingface-cli login
```

### Get HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create new token with **write** access
3. Copy token (starts with `hf_`)

---

## 💡 Tips

1. **Preprocess once, train many times**: After preprocessing, you can run multiple training experiments with different hyperparameters without re-tokenizing

2. **Use HuggingFace for team collaboration**: Upload preprocessed data once, let your entire team download and use it

3. **Name datasets clearly**: Use descriptive names like `username/model-dataset-tokenized` instead of `username/data`

4. **Private for sensitive data**: Use `--private` flag for proprietary or sensitive datasets

5. **Check disk space**: Preprocessed datasets can be 2-5x the original JSON size for vision-language models

6. **Test downloads**: Always test downloading your uploaded dataset to ensure it works correctly

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Install required packages: `pip install datasets huggingface_hub` |
| "Not authenticated" | Run `huggingface-cli login` |
| "Dataset not found" | Check repo_id format: `username/dataset-name` |
| "Permission denied" | Check token has write access for uploads |
| Upload is slow | Use `--max_shard_size` to split into smaller files |
| Out of disk space | Clean old preprocessed data or use larger storage |

---

## 📞 Need More Help?

- **Preprocessing issues**: See `PREPROCESSING_GUIDE.md`
- **HuggingFace issues**: See `HUGGINGFACE_GUIDE.md`
- **Quick start**: See `README_PREPROCESSING.md` or `QUICKSTART_HUGGINGFACE.md`
