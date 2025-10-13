"""Master training script for all transformer models."""

import sys
import argparse
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer
import wandb
import time
import json

# Import local modules
from utils import load_config, merge_configs, set_seed, get_device, count_parameters, format_time
from dataset import load_datasets
from model import load_model
from metrics import compute_metrics


def train(base_config_path, model_config_path, seed=42, run_name=None, wandb_disabled=False):
    """
    Train a transformer model.

    Args:
        base_config_path: Path to base config YAML
        model_config_path: Path to model config YAML
        seed: Random seed for reproducibility
        run_name: Optional custom name for wandb run
        wandb_disabled: If True, disable wandb logging

    Returns:
        dict: Final test results
    """
    start_time = time.time()

    # Load configs
    print("\n" + "="*70)
    print("LOADING CONFIGURATIONS")
    print("="*70)
    base_config = load_config(base_config_path)
    model_config = load_config(model_config_path)
    config = merge_configs(base_config, model_config)

    print(f"Base config: {base_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Random seed: {seed}")

    # Set seed for reproducibility
    set_seed(seed)

    # Get device
    device = get_device()

    # Initialize wandb
    if not wandb_disabled:
        wandb_config = {
            'model_name': config['model']['name'],
            'model_type': config['model']['type'],
            'seed': seed,
            'max_seq_length': config['preprocessing']['max_seq_length'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['per_device_train_batch_size'],
            'num_epochs': config['training']['num_train_epochs'],
        }

        if run_name is None:
            run_name = f"{config['model']['type']}-seed{seed}"

        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb'].get('entity'),
            name=run_name,
            config=wandb_config,
        )
        print(f"\n✓ Wandb initialized: {run_name}")
    else:
        print("\n⚠️  Wandb logging disabled")

    # Load datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)

    train_dataset, val_dataset, test_dataset = load_datasets(
        data_path=config['data']['raw_csv'],
        split_indices_path=config['data']['split_indices'],
        tokenizer_name=config['model']['tokenizer'],
        max_length=config['preprocessing']['max_seq_length'],
        cache_dir=config['data']['cache_dir']
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val:   {len(val_dataset):,}")
    print(f"  Test:  {len(test_dataset):,}")

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    model = load_model(config, num_labels=2)
    model.to(device)

    # Count parameters
    param_count = count_parameters(model)
    print(f"\nTrainable parameters: {param_count:,}")
    if not wandb_disabled:
        wandb.log({"parameter_count": param_count})

    # Create output directory
    output_dir = Path(config['training']['output_dir']) / f"{config['model']['type']}-seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Training arguments
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],

        # Optimization
        fp16=config['training']['fp16'] and torch.cuda.is_available(),
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],

        # Evaluation
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],

        # Saving
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],

        # Logging
        logging_steps=config['training']['logging_steps'],

        # Model selection
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],

        # Reproducibility
        seed=seed,

        # Wandb
        report_to="wandb" if not wandb_disabled else "none",
    )

    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Train batch size: {training_args.per_device_train_batch_size}")
    print(f"Eval batch size: {training_args.per_device_eval_batch_size}")
    print(f"FP16: {training_args.fp16}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Warmup ratio: {training_args.warmup_ratio}")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Model: {config['model']['type']}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    trainer.train()

    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    val_results = trainer.evaluate()

    print(f"\nValidation Results:")
    for key, value in val_results.items():
        if not key.startswith('eval_'):
            continue
        metric_name = key.replace('eval_', '')
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")

    if not wandb_disabled:
        wandb.log({"final_val_" + k: v for k, v in val_results.items()})

    # Evaluate on test set
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    test_results = trainer.evaluate(test_dataset)

    print(f"\nTest Results:")
    for key, value in test_results.items():
        if not key.startswith('eval_'):
            continue
        metric_name = key.replace('eval_', '')
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")

    if not wandb_disabled:
        wandb.log({"final_test_" + k: v for k, v in test_results.items()})

    # Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"\n✓ Model saved to: {final_model_path}")

    # Save results
    results_file = output_dir / "results.json"
    results_data = {
        'model_type': config['model']['type'],
        'model_name': config['model']['name'],
        'seed': seed,
        'val_results': val_results,
        'test_results': test_results,
        'training_config': {
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['per_device_train_batch_size'],
            'epochs': config['training']['num_train_epochs'],
            'max_seq_length': config['preprocessing']['max_seq_length'],
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"✓ Results saved to: {results_file}")

    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model: {config['model']['type']}")
    print(f"Seed: {seed}")
    print(f"Training time: {format_time(training_time)}")
    print(f"\nFinal Metrics:")
    print(f"  Val F1:  {val_results.get('eval_f1', 0):.4f}")
    print(f"  Test F1: {test_results.get('eval_f1', 0):.4f}")
    print(f"  Test FNR: {test_results.get('eval_fnr', 0):.4f} (False Negative Rate)")
    print("="*70 + "\n")

    if not wandb_disabled:
        wandb.finish()

    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer model for suicide ideation detection")
    parser.add_argument('--base_config', type=str, default='config/base_config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model configuration file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom name for wandb run')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    # Run training
    results = train(
        base_config_path=args.base_config,
        model_config_path=args.model_config,
        seed=args.seed,
        run_name=args.run_name,
        wandb_disabled=args.no_wandb
    )

    print("\n✅ Training completed successfully!")
