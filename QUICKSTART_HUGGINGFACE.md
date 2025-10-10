# Quick Start: Upload to HuggingFace

## 🎯 TL;DR

```bash
# 1. Login to HuggingFace (one-time)
huggingface-cli login

# 2. Upload your preprocessed dataset
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/my-dataset-name

# 3. Done! View at: https://huggingface.co/datasets/username/my-dataset-name
```

---

## 📤 Upload Preprocessed Dataset

### Step 1: Get HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "dataset-upload")
4. Select **Write** access
5. Copy the token (starts with `hf_`)

### Step 2: Login

```bash
huggingface-cli login
# Paste your token when prompted
```

### Step 3: Upload

**Public dataset:**
```bash
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-tokenized
```

**Private dataset:**
```bash
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope username/robospatial2-tokenized true
```

**Upload to organization:**
```bash
./upload_dataset.sh preprocessed_data/qwen2_5vl-3b-hope myorg/robospatial2-tokenized
```

---

## 📥 Download Preprocessed Dataset

```bash
# Download from HuggingFace
python download_from_huggingface.py \
    --repo_id username/robospatial2-tokenized \
    --output_path preprocessed_data/downloaded

# Use in training
llamafactory-cli train config.yaml --tokenized_path preprocessed_data/downloaded
```

---

## 🎨 Complete Workflow

```bash
# Machine 1: Preprocess and upload
./preprocess.sh                                    # Preprocess data
./upload_dataset.sh \                              # Upload to HuggingFace
    preprocessed_data/qwen2_5vl-3b-hope \
    username/robospatial2-tokenized

# Machine 2: Download and train
python download_from_huggingface.py \              # Download from HuggingFace
    --repo_id username/robospatial2-tokenized \
    --output_path preprocessed_data/tokenized

llamafactory-cli train config.yaml \               # Train with preprocessed data
    --tokenized_path preprocessed_data/tokenized
```

---

## ❓ Common Questions

**Q: Should I make it public or private?**
- Public: Good for research, open datasets, papers
- Private: Good for proprietary data, work-in-progress

**Q: How much storage do I get?**
- Free tier: Unlimited public datasets
- Free tier: Limited private datasets (check HuggingFace limits)

**Q: Can I update the dataset later?**
- Yes! Just re-run the upload script with the same name

**Q: How do I share with my team?**
- Public: Just share the link
- Private: Add them as collaborators in dataset settings

**Q: What if the dataset is too large?**
- Use `--max_shard_size` to split into smaller files
- Consider compressing before upload
- Use HuggingFace Pro for larger quotas

---

## 📚 Full Documentation

See [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md) for complete documentation.

---

## 🚀 Pro Tips

1. **Descriptive names**: Use clear names like `username/model-dataset-tokenized`
2. **Add README**: Document your dataset after uploading
3. **Use tags**: Add tags like "robotics", "vision-language", etc.
4. **Version control**: Use branches for different versions
5. **Test download**: Always test downloading before sharing

---

## 📝 Naming Convention

Good dataset names:
- ✅ `username/robospatial2-qwen25vl-3b-hope-tokenized`
- ✅ `username/vision-robotics-tokenized-v1`
- ✅ `myorg/project-model-data-preprocessed`

Avoid:
- ❌ `username/data`
- ❌ `username/test`
- ❌ `username/temp123`
