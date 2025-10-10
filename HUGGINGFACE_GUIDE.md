# HuggingFace Dataset Upload & Download Guide

This guide shows you how to upload and download preprocessed datasets to/from HuggingFace Hub.

## Why Upload to HuggingFace?

✅ **Share across machines** - Access your preprocessed data from any machine
✅ **Team collaboration** - Share with your team members
✅ **Backup** - Keep your preprocessed data safe in the cloud
✅ **Version control** - Track different versions of your preprocessed data
✅ **Reproducibility** - Share exact tokenized data with papers/projects

---

## 🚀 Upload Dataset to HuggingFace

### Prerequisites

1. **Get a HuggingFace account**: https://huggingface.co/join
2. **Create an access token**: https://huggingface.co/settings/tokens
   - Click "New token"
   - Select "Write" access
   - Copy the token

3. **Login to HuggingFace** (one-time setup):
```bash
huggingface-cli login
# Paste your token when prompted
```

### Method 1: Using the Shell Script (Easiest)

```bash
# Upload public dataset
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-qwen25vl-3b

# Upload private dataset
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-qwen25vl-3b true

# Upload to organization
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope myorg/robospatial2-qwen25vl-3b
```

### Method 2: Using Python Script (More Control)

```bash
# Basic upload (public)
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
    --repo_id username/robospatial2-qwen25vl-3b

# Upload private dataset
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
    --repo_id username/robospatial2-qwen25vl-3b \
    --private

# Upload with custom commit message
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
    --repo_id username/robospatial2-qwen25vl-3b \
    --commit_message "Updated with new data samples" \
    --max_shard_size 1GB

# Upload with specific token
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/qwen2_5vl-3b-hope \
    --repo_id username/robospatial2-qwen25vl-3b \
    --token hf_xxxxxxxxxxxxx
```

### What Gets Uploaded?

The entire preprocessed dataset directory including:
- All tokenized samples
- Dataset metadata
- Train/eval splits
- Any preprocessing settings

---

## 📥 Download Dataset from HuggingFace

### Method 1: Using Python Script

```bash
# Download public dataset
python download_from_huggingface.py \
    --repo_id username/robospatial2-qwen25vl-3b \
    --output_path preprocessed_data/downloaded

# Download private dataset
python download_from_huggingface.py \
    --repo_id username/robospatial2-qwen25vl-3b \
    --output_path preprocessed_data/downloaded \
    --token hf_xxxxxxxxxxxxx

# Download specific split only
python download_from_huggingface.py \
    --repo_id username/robospatial2-qwen25vl-3b \
    --output_path preprocessed_data/train_only \
    --split train
```

### Method 2: Using Python Code

```python
from datasets import load_dataset

# Download and save to disk
dataset = load_dataset("username/robospatial2-qwen25vl-3b")
dataset.save_to_disk("preprocessed_data/downloaded")

# For private datasets
dataset = load_dataset(
    "username/robospatial2-qwen25vl-3b",
    token="hf_xxxxxxxxxxxxx"
)
dataset.save_to_disk("preprocessed_data/downloaded")
```

### Use Downloaded Dataset in Training

After downloading, add to your config:

```yaml
tokenized_path: preprocessed_data/downloaded
```

Or use command line:

```bash
llamafactory-cli train config.yaml --tokenized_path preprocessed_data/downloaded
```

---

## 🔒 Public vs Private Datasets

### Public Dataset
- ✅ Anyone can download and use
- ✅ Good for research and open source
- ✅ Easier to share and cite
- ❌ Cannot restrict access

### Private Dataset
- ✅ Only you (and collaborators) can access
- ✅ Good for proprietary data
- ✅ Can control who has access
- ❌ Requires authentication to download

**Recommendation**: Use private for sensitive/proprietary data, public for research datasets.

---

## 📊 Managing Your Datasets on HuggingFace

### View Your Uploaded Dataset
```
https://huggingface.co/datasets/username/dataset-name
```

### Add Collaborators (Private Datasets)
1. Go to your dataset page
2. Click "Settings"
3. Add collaborators by username

### Update Dataset
Just re-run the upload script with the same repo_id. It will create a new commit.

### Delete Dataset
1. Go to dataset settings
2. Scroll down to "Danger Zone"
3. Click "Delete this dataset"

---

## 💡 Best Practices

### Naming Conventions

Use descriptive, clear names:
- ✅ `username/robospatial2-qwen25vl-3b-hope-tokenized`
- ✅ `username/dataset-model-config-tokenized`
- ❌ `username/data`
- ❌ `username/test123`

### Dataset Cards

After uploading, add a README to your dataset:
1. Go to your dataset page
2. Click "Edit dataset card"
3. Add information about:
   - What model it was tokenized for
   - Original dataset source
   - Preprocessing settings (cutoff_len, template, etc.)
   - How to use it

Example:
```markdown
# RoboSpatial2 Qwen2.5-VL-3B Tokenized Dataset

Preprocessed and tokenized version of the HOPE dataset for training Qwen2.5-VL-3B.

## Preprocessing Details
- Model: Qwen/Qwen2.5-VL-3B-Instruct
- Template: qwen2_vl
- Max sequence length: 2048
- Preprocessing workers: 42

## Usage
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("username/robospatial2-qwen25vl-3b-tokenized")
dataset.save_to_disk("local_path")

# Use in training config
tokenized_path: local_path
```

## Original Dataset
Based on the HOPE dataset for robotic spatial understanding.
```

### Version Control

Use branches or tags for different versions:
```bash
python upload_to_huggingface.py \
    --dataset_path preprocessed_data/v2 \
    --repo_id username/dataset \
    --branch v2.0
```

---

## 🔧 Troubleshooting

### "Failed to authenticate"
- Check your token at https://huggingface.co/settings/tokens
- Make sure token has "write" access
- Re-login with: `huggingface-cli login`

### "Repository not found" (when downloading)
- Check the repo_id is correct
- For private datasets, make sure you're logged in
- Verify you have access to the dataset

### Upload is slow
- Use `--max_shard_size` to split into smaller chunks
- Upload from a machine with better internet connection
- Consider compressing large datasets first

### Out of disk space
- Check available space before downloading
- Download specific splits only if needed
- Clean up old preprocessed data

---

## 📖 Full Workflow Example

```bash
# 1. Preprocess your data locally
./preprocess.sh

# 2. Upload to HuggingFace (private)
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-qwen25vl-3b true

# 3. On another machine, download the dataset
python download_from_huggingface.py \
    --repo_id username/robospatial2-qwen25vl-3b \
    --output_path preprocessed_data/qwen2_5vl-3b-hope

# 4. Use in training
llamafactory-cli train config.yaml \
    --tokenized_path preprocessed_data/qwen2_5vl-3b-hope
```

---

## 🌟 Advanced: Loading Directly from HuggingFace (No Download)

You can also load datasets directly from HuggingFace without downloading:

```python
from datasets import load_dataset

# This loads on-the-fly without saving to disk
dataset = load_dataset("username/robospatial2-qwen25vl-3b", split="train")
```

However, for training, it's recommended to download first for:
- Faster data loading during training
- Offline access
- Reduced network dependency

---

## 📞 Need Help?

- HuggingFace Docs: https://huggingface.co/docs/datasets
- HuggingFace Hub Guide: https://huggingface.co/docs/hub
- Create token: https://huggingface.co/settings/tokens
