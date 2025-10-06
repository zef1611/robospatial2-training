#!/usr/bin/env python3
"""
RoboSpatial2 Standalone Distributed Training Script

This script trains the Qwen2.5-VL model on the HOPE dataset without using LLaMA-Factory library.
It supports multi-node distributed training with DeepSpeed ZeRO-3.

Usage:
    # Single node:
    FORCE_TORCHRUN=1 torchrun --nproc_per_node=8 train_distributed_standalone.py

    # Multi-node (on each node):
    NODE_RANK=0 NNODES=2 MASTER_ADDR=tg10602 MASTER_PORT=29500 \
        FORCE_TORCHRUN=1 torchrun --nproc_per_node=8 \
        --nnodes=2 --node_rank=0 --master_addr=tg10602 --master_port=29500 \
        train_distributed_standalone.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
import yaml


# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    trust_remote_code: bool = True
    image_max_pixels: int = 262144
    video_max_pixels: int = 16384


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_dir: str = "data"
    dataset: str = "hope_train"
    eval_dataset: str = "hope_val"
    media_dir: str = "/scratch/h/huyle/dataset"
    cutoff_len: int = 2048
    max_samples: Optional[int] = None
    preprocessing_num_workers: int = 42
    dataloader_num_workers: int = 4


@dataclass
class FinetuningConfig:
    """Finetuning configuration."""
    freeze_vision_tower: bool = True
    freeze_multi_modal_projector: bool = True
    freeze_language_model: bool = False


class VisionLanguageDataset(Dataset):
    """Dataset for vision-language training in ShareGPT format."""

    def __init__(
        self,
        data_path: str,
        media_dir: str,
        processor: Any,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to JSON file with data
            media_dir: Directory containing images
            processor: Qwen2VL processor
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to use
        """
        self.media_dir = Path(media_dir)
        self.processor = processor
        self.max_length = max_length

        # Load data
        print(f"{YELLOW}Loading dataset from {data_path}...{NC}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[:max_samples]

        print(f"{GREEN}Loaded {len(self.data)} samples{NC}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        item = self.data[idx]
        messages = item['messages']
        images = item.get('images', [])

        # Format conversation for Qwen2VL
        conversation = []
        for msg in messages:
            role = msg['role']
            content = msg['content']

            # Handle images in user messages
            if role == 'user' and images:
                # Create content with image
                content_parts = []
                for img_path in images:
                    img_full_path = self.media_dir / img_path
                    if img_full_path.exists():
                        content_parts.append({
                            "type": "image",
                            "image": str(img_full_path)
                        })
                content_parts.append({
                    "type": "text",
                    "text": content
                })
                conversation.append({
                    "role": role,
                    "content": content_parts
                })
            else:
                conversation.append({
                    "role": role,
                    "content": content
                })

        # Apply chat template and process
        try:
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )

            # Process images
            image_inputs = None
            if images:
                pil_images = []
                for img_path in images:
                    img_full_path = self.media_dir / img_path
                    if img_full_path.exists():
                        pil_images.append(Image.open(img_full_path).convert('RGB'))

                if pil_images:
                    image_inputs = self.processor.image_processor(
                        images=pil_images,
                        return_tensors="pt"
                    )

            # Tokenize
            model_inputs = self.processor(
                text=text,
                images=image_inputs['pixel_values'] if image_inputs else None,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

            # Set labels (copy of input_ids for causal LM)
            model_inputs["labels"] = model_inputs["input_ids"].clone()

            # Squeeze batch dimension
            return {k: v.squeeze(0) for k, v in model_inputs.items()}

        except Exception as e:
            print(f"{RED}Error processing sample {idx}: {e}{NC}")
            # Return a dummy sample
            dummy = self.processor(
                text="Error",
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            dummy["labels"] = dummy["input_ids"].clone()
            return {k: v.squeeze(0) for k, v in dummy.items()}


def freeze_model_components(model, config: FinetuningConfig):
    """Freeze specific model components based on config."""
    print(f"\n{YELLOW}Freezing model components:{NC}")

    if config.freeze_vision_tower:
        if hasattr(model, 'visual'):
            for param in model.visual.parameters():
                param.requires_grad = False
            print(f"  ✓ Vision tower frozen")

    if config.freeze_multi_modal_projector:
        if hasattr(model, 'merger'):
            for param in model.merger.parameters():
                param.requires_grad = False
            print(f"  ✓ Multi-modal projector frozen")

    if config.freeze_language_model:
        if hasattr(model, 'model'):
            for param in model.model.parameters():
                param.requires_grad = False
            print(f"  ✓ Language model frozen")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{GREEN}Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%){NC}")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
    }


def main():
    """Main training function."""
    print(f"{GREEN}============================================================{NC}")
    print(f"{GREEN}RoboSpatial2 Standalone Distributed Training Script{NC}")
    print(f"{GREEN}============================================================{NC}")

    # Setup distributed training
    dist_info = setup_distributed()
    is_main_process = dist_info['rank'] == 0

    if is_main_process:
        print(f"\n{YELLOW}Distributed Info:{NC}")
        print(f"  Rank: {dist_info['rank']}")
        print(f"  World Size: {dist_info['world_size']}")
        print(f"  Local Rank: {dist_info['local_rank']}")

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    config_file = script_dir / "configs" / "yaml" / "Qwen2_5-VL-3B-Instruct.yaml"

    if is_main_process:
        print(f"\n{YELLOW}Paths:{NC}")
        print(f"  Script Dir: {script_dir}")
        print(f"  Config File: {config_file}")

    # Load configuration
    if not config_file.exists():
        print(f"\n{RED}Error: Config file not found at {config_file}{NC}")
        sys.exit(1)

    yaml_config = load_yaml_config(str(config_file))

    # Extract configurations
    model_name = yaml_config.get('model_name_or_path', 'Qwen/Qwen2.5-VL-3B-Instruct')
    dataset_dir = script_dir / yaml_config.get('dataset_dir', 'data')
    media_dir = yaml_config.get('media_dir', '/scratch/h/huyle/dataset')
    output_dir = yaml_config.get('output_dir', '/scratch/h/huyle/weights/robospatial2/qwen2_5vl-3b/full/sft')
    deepspeed_config = yaml_config.get('deepspeed')

    # Training arguments from YAML
    training_args_dict = {
        'output_dir': output_dir,
        'per_device_train_batch_size': yaml_config.get('per_device_train_batch_size', 48),
        'per_device_eval_batch_size': yaml_config.get('per_device_eval_batch_size', 48),
        'gradient_accumulation_steps': yaml_config.get('gradient_accumulation_steps', 2),
        'learning_rate': yaml_config.get('learning_rate', 2e-5),
        'num_train_epochs': yaml_config.get('num_train_epochs', 20.0),
        'lr_scheduler_type': yaml_config.get('lr_scheduler_type', 'cosine'),
        'warmup_ratio': yaml_config.get('warmup_ratio', 0.1),
        'bf16': yaml_config.get('bf16', True),
        'logging_steps': yaml_config.get('logging_steps', 5),
        'save_strategy': yaml_config.get('save_strategy', 'steps'),
        'save_steps': yaml_config.get('save_steps', 1),
        'evaluation_strategy': yaml_config.get('eval_strategy', 'epoch'),
        'save_total_limit': 3,
        'load_best_model_at_end': False,
        'report_to': yaml_config.get('report_to', 'wandb'),
        'run_name': yaml_config.get('run_name', 'rs2_qwen2_5vl-3b-sft'),
        'ddp_timeout': yaml_config.get('ddp_timeout', 180000000),
        'dataloader_num_workers': yaml_config.get('dataloader_num_workers', 4),
        'remove_unused_columns': False,
        'deepspeed': deepspeed_config,
        'gradient_checkpointing': True,
        'local_rank': dist_info['local_rank'],
    }

    # Resume from checkpoint if specified
    resume_checkpoint = yaml_config.get('resume_from_checkpoint')
    if resume_checkpoint and resume_checkpoint != 'null':
        training_args_dict['resume_from_checkpoint'] = resume_checkpoint

    if is_main_process:
        print(f"\n{YELLOW}Configuration:{NC}")
        print(f"  Model: {model_name}")
        print(f"  Dataset Dir: {dataset_dir}")
        print(f"  Media Dir: {media_dir}")
        print(f"  Output Dir: {output_dir}")
        print(f"  DeepSpeed: {deepspeed_config}")
        print(f"  Batch Size: {training_args_dict['per_device_train_batch_size']}")
        print(f"  Gradient Accumulation: {training_args_dict['gradient_accumulation_steps']}")
        print(f"  Learning Rate: {training_args_dict['learning_rate']}")
        print(f"  Epochs: {training_args_dict['num_train_epochs']}")

    # Load processor and tokenizer
    if is_main_process:
        print(f"\n{YELLOW}Loading processor and tokenizer...{NC}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer = processor.tokenizer

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print(f"{GREEN}Processor and tokenizer loaded{NC}")

    # Load model
    if is_main_process:
        print(f"\n{YELLOW}Loading model...{NC}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args_dict['bf16'] else torch.float32,
        device_map={'': dist_info['local_rank']},
    )

    if is_main_process:
        print(f"{GREEN}Model loaded{NC}")

    # Freeze components
    finetuning_config = FinetuningConfig(
        freeze_vision_tower=yaml_config.get('freeze_vision_tower', True),
        freeze_multi_modal_projector=yaml_config.get('freeze_multi_modal_projector', True),
        freeze_language_model=yaml_config.get('freeze_language_model', False),
    )

    if is_main_process:
        freeze_model_components(model, finetuning_config)

    # Load datasets
    if is_main_process:
        print(f"\n{YELLOW}Loading datasets...{NC}")

    train_dataset_name = yaml_config.get('dataset', 'hope_train')
    eval_dataset_name = yaml_config.get('eval_dataset', 'hope_val')
    max_samples = yaml_config.get('max_samples')

    train_data_path = dataset_dir / f"{train_dataset_name}.json"
    eval_data_path = dataset_dir / f"{eval_dataset_name}.json"

    train_dataset = VisionLanguageDataset(
        data_path=str(train_data_path),
        media_dir=media_dir,
        processor=processor,
        max_length=yaml_config.get('cutoff_len', 2048),
        max_samples=max_samples,
    )

    eval_dataset = VisionLanguageDataset(
        data_path=str(eval_data_path),
        media_dir=media_dir,
        processor=processor,
        max_length=yaml_config.get('cutoff_len', 2048),
        max_samples=None,  # Use all validation data
    )

    if is_main_process:
        print(f"{GREEN}Datasets loaded:{NC}")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Eval: {len(eval_dataset)} samples")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    # Training arguments
    training_args = TrainingArguments(**training_args_dict)

    # Create trainer
    if is_main_process:
        print(f"\n{YELLOW}Creating trainer...{NC}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if is_main_process:
        print(f"{GREEN}Trainer created{NC}")

    # Check for existing checkpoints
    last_checkpoint = None
    if os.path.isdir(output_dir) and not training_args_dict.get('resume_from_checkpoint'):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            if is_main_process:
                print(f"{YELLOW}Found checkpoint: {last_checkpoint}{NC}")

    # Start training
    if is_main_process:
        print(f"\n{GREEN}============================================================{NC}")
        print(f"{GREEN}Starting training...{NC}")
        print(f"{GREEN}============================================================{NC}\n")

    try:
        checkpoint = training_args_dict.get('resume_from_checkpoint') or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Save model
        if is_main_process:
            print(f"\n{YELLOW}Saving model...{NC}")
        trainer.save_model()

        # Save metrics
        if is_main_process:
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

        if is_main_process:
            print(f"\n{GREEN}============================================================{NC}")
            print(f"{GREEN}Training completed successfully!{NC}")
            print(f"{GREEN}============================================================{NC}")
            print(f"\nModel saved to: {output_dir}")
            print(f"\nTraining metrics:")
            for key, value in train_result.metrics.items():
                print(f"  {key}: {value}")

    except Exception as e:
        if is_main_process:
            print(f"\n{RED}============================================================{NC}")
            print(f"{RED}Training failed!{NC}")
            print(f"{RED}Error: {e}{NC}")
            print(f"{RED}============================================================{NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
