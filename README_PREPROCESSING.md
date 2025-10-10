# Quick Start: Pre-tokenize Your Data

## The Problem
Data preprocessing (tokenization) takes too long at the start of each training run.

## The Solution
Tokenize once, train many times! 🚀

## Steps

### 1️⃣ Preprocess your data (one time only)

```bash
cd robospatial2-training
./preprocess.sh
```

This creates tokenized data in `preprocessed_data/qwen2_5vl-3b-hope/`

### 2️⃣ Use the preprocessed data in training

**Option A**: Use the example config (recommended)
```bash
llamafactory-cli train configs/yaml/Qwen2_5-VL-3B-Instruct-preprocessed.yaml
```

**Option B**: Add to your existing config
```yaml
tokenized_path: /project/6100855/huyle/projects/robospatial2/robospatial2-training/preprocessed_data/qwen2_5vl-3b-hope
```

**Option C**: Command line override
```bash
llamafactory-cli train configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --tokenized_path preprocessed_data/qwen2_5vl-3b-hope
```

### 3️⃣ Train with instant startup! ⚡

```bash
./train.sh  # Uses the config with tokenized_path
```

## What You Get

✅ **No more waiting** for data preprocessing at training startup
✅ **Consistent tokenization** across all training runs
✅ **Run multiple experiments** without re-tokenizing
✅ **Easy debugging** - preprocess once, check once

## Files Created

| File | Purpose |
|------|---------|
| `preprocess.sh` | Easy-to-use shell script for preprocessing |
| `preprocess_data.py` | Python script that does the actual tokenization |
| `upload_dataset.sh` | Shell script to upload to HuggingFace |
| `upload_to_huggingface.py` | Python script for HuggingFace upload |
| `download_from_huggingface.py` | Python script to download from HuggingFace |
| `configs/yaml/Qwen2_5-VL-3B-Instruct-preprocessed.yaml` | Example config using preprocessed data |
| `PREPROCESSING_GUIDE.md` | Detailed preprocessing documentation |
| `HUGGINGFACE_GUIDE.md` | Complete HuggingFace upload/download guide |
| `QUICKSTART_HUGGINGFACE.md` | Quick start for HuggingFace |

## When to Re-preprocess

Re-run preprocessing if you change:
- Model or tokenizer
- Dataset content
- Prompt template
- Max sequence length (`cutoff_len`)

## Custom Usage

Different config:
```bash
./preprocess.sh configs/yaml/MyConfig.yaml preprocessed_data/my-output
```

Specific datasets:
```bash
python preprocess_data.py \
    --config configs/yaml/Qwen2_5-VL-3B-Instruct.yaml \
    --output preprocessed_data/custom \
    --dataset hope_train \
    --eval-dataset hope_val
```

## 🌐 Share via HuggingFace (Optional)

Want to share your preprocessed data across machines or with your team?

### Upload to HuggingFace:
```bash
# One-time setup
huggingface-cli login

# Upload your preprocessed data
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/my-dataset-name
```

### Download on another machine:
```bash
python download_from_huggingface.py \
    --repo_id username/my-dataset-name \
    --output_path preprocessed_data/downloaded
```

See [QUICKSTART_HUGGINGFACE.md](QUICKSTART_HUGGINGFACE.md) for quick guide or [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md) for full documentation.

## Need Help?

- **Preprocessing**: See `PREPROCESSING_GUIDE.md` for detailed documentation and troubleshooting
- **HuggingFace Upload/Download**: See `QUICKSTART_HUGGINGFACE.md` or `HUGGINGFACE_GUIDE.md`

---

**Pro Tip**: After preprocessing, you can experiment with different learning rates, batch sizes, and other training hyperparameters without ever re-tokenizing! 🎯
