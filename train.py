#!/usr/bin/env python
"""
Training script for DinoMHC MHC-Peptide Binding Prediction.

Usage:
    # Train with default config
    python train.py --config configs/default.yaml --fold 0
    
    # Train all 5 folds
    python train.py --config configs/default.yaml --train_all_folds
    
    # Override config values via CLI
    python train.py --config configs/default.yaml --fold 0 --training.batch_size 32
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from src.data_module import MHCPeptideDataModule
from src.lightning_module import create_lightning_module


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def flatten_config(config: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested config dict."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_config(flat_config: Dict, sep: str = '.') -> Dict:
    """Unflatten a flat config dict back to nested structure."""
    result = {}
    for key, value in flat_config.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DinoMHC for MHC-Peptide Binding Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    
    # Fold selection
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Fold index to use (0-4)"
    )
    parser.add_argument(
        "--train_all_folds", action="store_true",
        help="Train on all 5 folds sequentially"
    )
    parser.add_argument(
        "--resume_from_checkpoint", action='store_true',
        help="Path to checkpoint to resume training from"
    ) # Remember that this argument will search and use the latest fold in the checkpoint path to continue training
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    # Allow overriding any config value via CLI using dot notation
    # e.g., --training.batch_size 32 --model.dim 256
    args, unknown = parser.parse_known_args()
    
    # Parse unknown args as config overrides
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]  # Remove --
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                # Try to parse as int, float, bool, or keep as string
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'null' or value.lower() == 'none':
                            value = None
                overrides[key] = value
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1
    
    return args, overrides


def get_model_config(config: Dict) -> Dict[str, Any]:
    """Build model configuration from config dict."""
    return {
        'dim': config['model']['dim'],
        'num_groove_tokens': config['model']['num_groove_tokens'],
        'num_fusion_layers': config['model']['num_fusion_layers'],
        'num_geometry_blocks': config['model']['num_geometry_blocks'],
        'num_heads': config['model']['num_heads'],
        'dropout': config['model']['dropout'],
        'max_peptide_length': config['data']['max_peptide_length'],
        'task_head': config['model']['task_head'],
        'use_flanks': config['data']['use_flanks'],
        'flank_pooling': config['model']['flank_pooling'],
        'encoder_type': config['model']['encoder_type'],
        'esm_model_name': config['esm']['model_name'],
        'esm_unfreeze_layers': config['esm']['unfreeze_layers'],
        'esm_unfreeze_embeddings': config['esm']['unfreeze_embeddings'],
    }


def create_callbacks(config: Dict, fold: int) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Checkpoint callback - save best models by AUPRC
    checkpoint_dir = Path(config['logging']['output_dir']) / config['logging']['experiment_name'] / f"fold_{fold}" / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch={epoch:02d}-auprc={val/auprc:.4f}",
        monitor="val/auprc",
        mode="max",
        save_top_k=config['callbacks']['save_top_k'],
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping based on AUPRC
    early_stopping = EarlyStopping(
        monitor="val/auprc",
        mode="max",
        patience=config['callbacks']['patience'],
        min_delta=0.001,
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Rich progress bar
    try:
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
    except (ImportError, Exception):
        pass
    
    return callbacks


def create_logger(config: Dict, fold: int):
    """Create logger(s) for training."""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['logging']['output_dir'],
        name=config['logging']['experiment_name'],
        version=f"fold_{fold}",
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    # WandB logger
    if config['logging']['use_wandb']:
        try:
            wandb_logger = WandbLogger(
                project=config['logging']['wandb_project'],
                entity=config['logging']['wandb_entity'],
                name=f"{config['logging']['experiment_name']}_fold_{fold}",
                save_dir=config['logging']['output_dir'],
                log_model=True,
            )
            loggers.append(wandb_logger)
        except Exception as e:
            print(f"Warning: Could not initialize WandB logger: {e}")
    
    return loggers

def get_current_epoch(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'epoch' in ckpt:
        return ckpt['epoch']
    else:
        return None

def train_fold(config: Dict, fold: int, verbose: bool = True, resume_from_checkpoint: bool = False):
    """Train a single fold."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Fold {fold}")
        print(f"{'='*60}\n")
    
    # Set seed for reproducibility
    pl.seed_everything(config['misc']['seed'] + fold, workers=True)
    
    # Set matmul precision for tensor cores
    torch.set_float32_matmul_precision('medium')
    
    # Create data module
    encoder_type = config['model']['encoder_type']
    tokenizer_type = 'esm2' if encoder_type in ['esm2', 'esm2_shared'] else 'embedding'
    
    data_module = MHCPeptideDataModule(
        data_dir=config['data']['data_dir'],
        fold=fold,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        tokenizer_type=tokenizer_type,
        esm_model_name=config['esm']['model_name'],
        max_peptide_length=config['data']['max_peptide_length'],
        max_mhc_length=config['data']['max_mhc_length'],
        use_flanks=config['data']['use_flanks'],
        flank_length=config['data']['flank_length'],
        binarize_labels=True,
        label_threshold=0.5,
    )
    
    # Setup to get dataset sizes
    data_module.setup('fit')
    
    # Calculate max_steps if not specified
    max_steps_cfg = config['training']['max_steps']
    if max_steps_cfg == -1:
        steps_per_epoch = len(data_module.train_dataloader())
        max_steps = steps_per_epoch * config['training']['max_epochs']
    else:
        max_steps = max_steps_cfg
    
    # Create model config
    model_config = get_model_config(config)
    
    # Create lightning module
    module_kwargs = {
        'config': model_config,
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_steps': config['training']['warmup_steps'],
        'max_steps': max_steps,
        'encoder_lr_factor': config['training']['encoder_lr_factor'],
        'pos_weight': config['loss'].get('pos_weight', None),
        'label_smoothing': config['loss']['label_smoothing'],
    }

    use_allele_balanced_loss = config['loss'].get('use_allele_balanced_loss', False)
    if use_allele_balanced_loss:
        module_kwargs['focal_alpha'] = config['loss']['focal_alpha']
        module_kwargs['focal_gamma'] = config['loss']['focal_gamma']
        module_kwargs['allele_weight_type'] = config['loss']['allele_weight_type']
        module_kwargs['allele_beta'] = config['loss'].get('allele_beta', 0.9999)
        module_kwargs['normalize_allele_weights'] = config['loss']['normalize_allele_weights']

    model = create_lightning_module(
        use_allele_balanced_loss=use_allele_balanced_loss,
        **module_kwargs
    )
    
    if verbose:
        # Print model info
        print("Model Configuration:")
        print("-" * 40)
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        print()
        
        # Print dataset info
        print("Dataset Info:")
        print("-" * 40)
        print(f"  Train samples: {data_module.train_dataset_size:,}")
        print(f"  Val samples: {data_module.val_dataset_size:,}")
        print(f"  Steps per epoch: {len(data_module.train_dataloader())}")
        print(f"  Max steps: {max_steps:,}")
        print()
    
    # Create callbacks and loggers
    callbacks = create_callbacks(config, fold)
    loggers = create_logger(config, fold)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        max_steps=max_steps_cfg if max_steps_cfg > 0 else -1,
        accelerator="gpu" if config['hardware']['gpus'] > 0 else "cpu",
        devices=config['hardware']['gpus'] if config['hardware']['gpus'] > 0 else "auto",
        strategy=config['hardware']['strategy'],
        precision=config['training']['precision'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        deterministic=True,
        log_every_n_steps=50,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
    )
    
    # Train
    if verbose:
        print("Starting training...")
    
    if resume_from_checkpoint:
        ckpt_path = Path(config['logging']['output_dir']) / config['logging']['experiment_name'] / f"fold_{fold}" / "checkpoints" / "last.ckpt"
        if ckpt_path.exists():
            current_epoch = get_current_epoch(str(ckpt_path))
            if current_epoch is not None and current_epoch < config['training']['max_epochs']:
                if verbose:
                    print(f"Resuming training from checkpoint: {ckpt_path} at epoch {current_epoch}")
                trainer.fit(model, data_module, ckpt_path=ckpt_path)
            else:
                if verbose:
                    print(f"Checkpoint epoch {current_epoch} >= max_epochs {config['training']['max_epochs']}. Starting fresh training.")
                return {
                    'fold': fold,
                    'best_ckpt': str(ckpt_path),
                    'best_auprc': None,
                }
    else:
        trainer.fit(model, data_module)
    
    # Get best checkpoint path and score
    best_ckpt = callbacks[0].best_model_path
    best_auprc = float(callbacks[0].best_model_score) if callbacks[0].best_model_score else None
    
    if verbose:
        print(f"\nBest checkpoint: {best_ckpt}")
        print(f"Best AUPRC: {best_auprc:.4f}" if best_auprc else "Best AUPRC: N/A")
    
    return {
        'fold': fold,
        'best_ckpt': best_ckpt,
        'best_auprc': best_auprc,
    }


def main():
    """Main training function."""
    args, overrides = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    flat_config = flatten_config(config)
    flat_config.update(overrides)
    config = unflatten_config(flat_config)
    
    # Print configuration
    print("\n" + "="*60)
    print("DinoMHC Training")
    print("="*60)
    print(f"\nConfig file: {args.config}")
    print(f"Experiment: {config['logging']['experiment_name']}")
    print(f"Encoder type: {config['model']['encoder_type']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    if config['loss'].get('use_allele_balanced_loss', False):
        print(f"Loss: Allele-Balanced Focal Loss")
        print(f"  - Focal: alpha={config['loss']['focal_alpha']}, gamma={config['loss']['focal_gamma']}")
        print(f"  - Allele weighting: {config['loss']['allele_weight_type']}")
        if config['loss']['allele_weight_type'] == 'effective_num':
            print(f"  - Beta: {config['loss'].get('allele_beta', 0.9999)}")
    else:
        print(f"Loss: BCE" + (f" (pos_weight={config['loss'].get('pos_weight')})" if config['loss'].get('pos_weight') else ""))
    print()
    
    # Create output directory
    output_path = Path(config['logging']['output_dir']) / config['logging']['experiment_name']
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = output_path / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")
    
    # Train
    results = []
    
    if args.train_all_folds:
        for fold in range(5):
            print('ARGS:', args.resume_from_checkpoint)
            result = train_fold(config, fold, args.verbose, args.resume_from_checkpoint)
            results.append(result)
        
        # Print summary
        print("\n" + "="*60)
        print("Training Complete - Summary")
        print("="*60)
        valid_results = [r for r in results if r['best_auprc'] is not None]
        for r in results:
            auprc_str = f"{r['best_auprc']:.4f}" if r['best_auprc'] else "N/A"
            print(f"  Fold {r['fold']}: AUPRC = {auprc_str}")
        
        if valid_results:
            avg_auprc = sum(r['best_auprc'] for r in valid_results) / len(valid_results)
            print(f"\n  Average AUPRC: {avg_auprc:.4f}")
    else:
        print('ARGS:', args.resume_from_checkpoint)
        result = train_fold(config, args.fold, args.verbose, args.resume_from_checkpoint)
        results.append(result)
    
    # Save results
    results_path = output_path / "results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
