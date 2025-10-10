# RoboSpatial2 Training

This directory contains scripts and configuration for training the Qwen2.5-VL model on the HOPE dataset for robotic spatial understanding tasks.

## Overview

The training pipeline fine-tunes the Qwen2.5-VL-7B-Instruct model on vision-language tasks related to:
- Camera extrinsics and pose estimation
- Object pose estimation (position, rotation, scale)
- 3D spatial understanding in robotic contexts

## Dataset

- **Training samples**: 100,392
- **Validation samples**: 34,015
- **Format**: Vision-language conversations with images
- **Source**: HOPE (Household Object Pose Estimation) dataset

## Files

- `convert_dataset.py` - Converts HOPE JSONL format to training JSON format
- `train_robospatial2.yaml` - Training configuration
- `train.sh` - Main training script
- `eval.sh` - Evaluation script
- `README.md` - This file

## Setup

1. Install llamafactory CLI:
   ```bash
   pip install llamafactory
   ```

2. (Optional) Install and setup Weights & Biases for experiment tracking:
   ```bash
   pip install wandb
   wandb login
   ```
   Note: By default, wandb is disabled. To enable it, change `report_to: none` to `report_to: wandb` in `train_robospatial2.yaml`.

3. Convert the dataset (if not already done):
   ```bash
   python convert_dataset.py
   ```

4. Datasets will be automatically registered in `robospatial2-training/data/dataset_info.json`

## Training

### Quick Start

```bash
./train.sh
```

### Training Configuration

Key settings in `train_robospatial2.yaml`:

- **Model**: Qwen2.5-VL-7B-Instruct
- **Method**: Full parameter fine-tuning with frozen vision tower
- **Batch size**: 1 per device
- **Gradient accumulation**: 2 steps
- **Learning rate**: 1e-5
- **Epochs**: 3
- **Optimizer**: AdamW with cosine LR schedule
- **Precision**: BF16
- **DeepSpeed**: ZeRO-3 optimization

### Hardware Requirements

- **GPU**: 1+ NVIDIA GPUs with 24GB+ VRAM (recommended)
- **DeepSpeed ZeRO-3** is used for memory-efficient training
- **Multi-GPU**: Automatically detected and used if available

### Monitoring

Training metrics are tracked with Weights & Biases:
- **Project**: `robospatial2`
- **Run name**: `robospatial2-qwen2_5vl-7b-sft`
- **Dashboard**: View real-time metrics at https://wandb.ai

Tracked metrics include:
- Training/validation loss
- Learning rate schedule
- GPU utilization
- Evaluation metrics every 1000 steps
- Model checkpoints every 1000 steps

Local logs are also saved in the output directory with loss curves.

## Evaluation

Run evaluation on validation set:

```bash
./eval.sh
```

Or manually:

```bash
cd robospatial2-training
llamafactory-cli eval train_robospatial2.yaml
```

## Output

Trained model and checkpoints are saved to:
```
robospatial2-training/saves/robospatial2-qwen2_5vl-7b/full/sft/
```

## Customization

To modify training parameters, edit `train_robospatial2.yaml`:

- `per_device_train_batch_size`: Adjust based on GPU memory
- `gradient_accumulation_steps`: Increase for larger effective batch size
- `learning_rate`: Tune for convergence
- `num_train_epochs`: More epochs for better performance
- `save_steps`: Checkpoint frequency
- `eval_steps`: Evaluation frequency
- `run_name`: Change W&B run name (also update `WANDB_RUN_NAME` in train.sh)
- `report_to`: Change to `none` to disable W&B tracking

To change the W&B project name, edit the `WANDB_PROJECT` environment variable in `train.sh`.

## Inference

After training, run inference:

```bash
./inference.sh
```

Or manually:

```bash
cd robospatial2-training
llamafactory-cli chat train_robospatial2.yaml
```

Or use the trained model in your own code by loading from the checkpoint.

## Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Ensure DeepSpeed ZeRO-3 is enabled
- Use smaller image resolution in config

### Slow Training

- Increase `dataloader_num_workers` if CPU is underutilized
- Enable mixed precision training (bf16/fp16)
- Use multiple GPUs if available

### Dataset Errors

- Verify image paths in JSON files are correct
- Check that all images exist in the dataset
- Re-run `convert_dataset.py` if needed

NNODES=2 NODE_RANK=0 MASTER_ADDR=tg10604 MASTER_PORT=29500 bash train_distributed.sh
NNODES=2 NODE_RANK=1 MASTER_ADDR=tg10604 MASTER_PORT=29500 bash train_distributed.sh


  1. On the master node (rank 0):
NNODES=4 NODE_RANK=0 MASTER_ADDR=tg10601 MASTER_PORT=29500 bash train_distributed.sh

  2. On worker node 1 (rank 1):
NNODES=4 NODE_RANK=1 MASTER_ADDR=tg10601 MASTER_PORT=29500 bash train_distributed.sh

  3. On worker node 2 (rank 2):
NNODES=4 NODE_RANK=2 MASTER_ADDR=tg10601 MASTER_PORT=29500 bash train_distributed.sh

  4. On worker node 3 (rank 3):
NNODES=4 NODE_RANK=3 MASTER_ADDR=tg10601 MASTER_PORT=29500 bash train_distributed.sh

  Key points:
  - Set NNODES=4 on all nodes
  - Set NODE_RANK to 0, 1, 2, 3 for each respective node
  - Set MASTER_ADDR to the hostname/IP of your master node (currently tg10602)
  - Use the same MASTER_PORT across all nodes (default 29500)
  - Ensure all nodes can communicate with each other over the network
  - Launch the script on all nodes simultaneously (or within a short time window)

## Citation

If you use this training setup, please cite:

```
RoboSpatial2: Fine-tuning Qwen2.5-VL for Robotic Spatial Understanding
```