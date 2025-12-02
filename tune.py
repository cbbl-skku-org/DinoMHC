#!/usr/bin/env python
"""
Hyperparameter Tuning for DinoMHC using Optuna.

This script optimizes hyperparameters based on 5-fold cross-validation AUPRC.
Supports parallel training on multiple GPUs.

Usage:
    # Run hyperparameter tuning (sequential folds)
    python tune.py --config configs/default.yaml --n_trials 100
    
    # Run with parallel folds on multiple GPUs (e.g., 5 folds on 5 GPUs)
    python tune.py --config configs/default.yaml --n_trials 100 --parallel_folds --n_gpus 5
    
    # Run multiple Optuna workers in parallel (requires shared storage)
    # Terminal 1:
    python tune.py --config configs/default.yaml --storage sqlite:///optuna.db --study_name my_study --n_trials 50
    # Terminal 2:
    python tune.py --config configs/default.yaml --storage sqlite:///optuna.db --study_name my_study --n_trials 50
    
    # Resume tuning from existing study
    python tune.py --config configs/default.yaml --study_name my_study --n_trials 50
"""

import os
import sys
import argparse
import yaml
import copy
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from lightning.pytorch.loggers import TensorBoardLogger

from src.data_module import MHCPeptideDataModule
from src.lightning_module import create_lightning_module


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
        'task_head': 'presentation',
        'encoder_type': config['model']['encoder_type'],
        'esm_model_name': config['esm']['model_name'],
        'esm_unfreeze_layers': config['esm']['unfreeze_layers'],
        'esm_unfreeze_embeddings': config['esm']['unfreeze_embeddings'],
    }


def suggest_hyperparameters(trial: Trial, base_config: Dict) -> Dict:
    """Suggest hyperparameters for this trial."""
    config = copy.deepcopy(base_config)
    
    # Model hyperparameters
    config['model']['dim'] = trial.suggest_categorical('dim', [128, 256, 320])
    config['model']['num_groove_tokens'] = trial.suggest_int('num_groove_tokens', 30, 60, step=10)
    config['model']['num_fusion_layers'] = trial.suggest_int('num_fusion_layers', 2, 4)
    config['model']['num_geometry_blocks'] = trial.suggest_int('num_geometry_blocks', 1, 2)
    config['model']['num_heads'] = trial.suggest_categorical('num_heads', [4, 8])
    config['model']['dropout'] = trial.suggest_float('dropout', 0.05, 0.3, step=0.05)
    
    # Training hyperparameters
    config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    config['training']['warmup_steps'] = trial.suggest_int('warmup_steps', 500, 3000, step=500)
    config['training']['encoder_lr_factor'] = trial.suggest_float('encoder_lr_factor', 0.01, 0.5, log=True)
    
    # Loss hyperparameters - Allele-Balanced Focal Loss
    config['loss']['use_allele_balanced_loss'] = trial.suggest_categorical('use_allele_balanced_loss', [True])

    if config['loss']['use_allele_balanced_loss']:
        # Focal loss parameters (for class imbalance)
        config['loss']['focal_alpha'] = trial.suggest_float('focal_alpha', 0.1, 0.5, step=0.05)
        config['loss']['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0, step=0.5)

        # Allele balancing parameters
        config['loss']['allele_weight_type'] = trial.suggest_categorical('allele_weight_type', ['inverse_freq'])
        if config['loss']['allele_weight_type'] == 'effective_num':
            config['loss']['allele_beta'] = trial.suggest_float('allele_beta', 0.99, 0.9999, log=True)
        config['loss']['normalize_allele_weights'] = trial.suggest_categorical('normalize_allele_weights', [True, False])
    else:
        # Standard BCE with pos_weight
        use_pos_weight = trial.suggest_categorical('use_pos_weight', [True, False])
        if use_pos_weight:
            config['loss']['pos_weight'] = trial.suggest_float('pos_weight', 5.0, 20.0, step=1.0)

    config['loss']['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.1, step=0.02)
    
    return config


def train_fold_for_tuning(
    config: Dict,
    fold: int,
    trial_number: int,
    output_dir: Path,
    max_epochs: int = 10,
    gpu_id: int = 0,
) -> Optional[float]:
    """Train a single fold for hyperparameter tuning.
    
    Args:
        config: Configuration dictionary
        fold: Fold number
        trial_number: Optuna trial number (for logging)
        output_dir: Output directory
        max_epochs: Maximum epochs
        gpu_id: GPU device ID to use
    """
    
    # Set seed
    pl.seed_everything(config['misc']['seed'] + fold, workers=True)
    torch.set_float32_matmul_precision('medium')
    
    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    
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
    
    data_module.setup('fit')
    
    # Calculate max_steps
    steps_per_epoch = len(data_module.train_dataloader())
    max_steps = steps_per_epoch * max_epochs
    
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

    try:
        model = create_lightning_module(
            use_allele_balanced_loss=use_allele_balanced_loss,
            **module_kwargs
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        return None
    
    # Callbacks
    checkpoint_dir = output_dir / f"trial_{trial_number}" / f"fold_{fold}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor="val/auprc",
        mode="max",
        save_top_k=1,
        verbose=False,
    )
    
    early_stopping = EarlyStopping(
        monitor="val/auprc",
        mode="max",
        patience=5,  # Shorter patience for tuning
        min_delta=0.001,
        verbose=False,
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="optuna_trials",
        version=f"trial_{trial_number}_fold_{fold}",
        default_hp_metric=False,
    )
    
    # Trainer - use specific GPU
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[gpu_id] if torch.cuda.is_available() else "auto",
        strategy="auto",  # Single GPU per fold
        precision=config['training']['precision'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=True,
        log_every_n_steps=100,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
    )
    
    try:
        trainer.fit(model, data_module)
        best_auprc = float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None
        return best_auprc
    except Exception as e:
        print(f"Error during training fold {fold}: {e}")
        return None


def train_fold_worker(args: Tuple) -> Tuple[int, Optional[float]]:
    """Worker function for parallel fold training.
    
    Args:
        args: Tuple of (config, fold, trial_number, output_dir, max_epochs, gpu_id)
    
    Returns:
        Tuple of (fold, auprc)
    """
    config, fold, trial_number, output_dir, max_epochs, gpu_id = args
    auprc = train_fold_for_tuning(
        config=config,
        fold=fold,
        trial_number=trial_number,
        output_dir=Path(output_dir),
        max_epochs=max_epochs,
        gpu_id=gpu_id,
    )
    return fold, auprc


def train_folds_parallel(
    config: Dict,
    trial_number: int,
    output_dir: Path,
    max_epochs: int,
    n_folds: int,
    n_gpus: int,
) -> List[Optional[float]]:
    """Train multiple folds in parallel on multiple GPUs.

    Args:
        config: Configuration dictionary
        trial_number: Optuna trial number
        output_dir: Output directory
        max_epochs: Maximum epochs per fold
        n_folds: Number of folds
        n_gpus: Number of available GPUs

    Returns:
        List of AUPRC scores for each fold
    """
    # Run folds in parallel using multiprocessing
    # Use spawn to avoid CUDA issues with fork
    ctx = mp.get_context('spawn')

    fold_auprcs = [None] * n_folds

    # Process folds in batches equal to n_gpus to ensure balanced GPU utilization
    # This prevents GPU 0 from being overloaded when n_folds > n_gpus
    for batch_start in range(0, n_folds, n_gpus):
        batch_end = min(batch_start + n_gpus, n_folds)
        batch_folds = list(range(batch_start, batch_end))

        # Assign each fold in this batch to a unique GPU
        worker_args = [
            (config, fold, trial_number, str(output_dir), max_epochs, i)
            for i, fold in enumerate(batch_folds)
        ]

        print(f"    Processing folds {batch_folds} on GPUs {list(range(len(batch_folds)))}")

        # Run this batch of folds in parallel
        with ProcessPoolExecutor(max_workers=len(batch_folds), mp_context=ctx) as executor:
            futures = {executor.submit(train_fold_worker, args): args[1] for args in worker_args}

            for future in as_completed(futures):
                fold = futures[future]
                try:
                    result_fold, auprc = future.result()
                    fold_auprcs[result_fold] = auprc
                    print(f"    Fold {result_fold} completed: AUPRC = {auprc:.4f}" if auprc else f"    Fold {result_fold} FAILED")
                except Exception as e:
                    import traceback
                    print(f"    Fold {fold} raised exception: {e}")
                    traceback.print_exc()
                    fold_auprcs[fold] = None

        # Clean up GPU memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fold_auprcs


def objective(
    trial: Trial,
    base_config: Dict,
    output_dir: Path,
    max_epochs: int,
    n_folds: int,
    parallel_folds: bool = False,
    n_gpus: int = 1,
) -> float:
    """Optuna objective function: 5-fold CV average AUPRC."""
    
    # Suggest hyperparameters
    config = suggest_hyperparameters(trial, base_config)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print("Hyperparameters:")
    print(f"  dim: {config['model']['dim']}")
    print(f"  num_groove_tokens: {config['model']['num_groove_tokens']}")
    print(f"  num_fusion_layers: {config['model']['num_fusion_layers']}")
    print(f"  num_geometry_blocks: {config['model']['num_geometry_blocks']}")
    print(f"  dropout: {config['model']['dropout']}")
    print(f"  learning_rate: {config['training']['learning_rate']:.2e}")
    print(f"  weight_decay: {config['training']['weight_decay']:.2e}")
    print(f"  use_allele_balanced_loss: {config['loss'].get('use_allele_balanced_loss', False)}")
    if config['loss'].get('use_allele_balanced_loss', False):
        print(f"  focal_alpha: {config['loss']['focal_alpha']}")
        print(f"  focal_gamma: {config['loss']['focal_gamma']}")
        print(f"  allele_weight_type: {config['loss']['allele_weight_type']}")
        if config['loss']['allele_weight_type'] == 'effective_num':
            print(f"  allele_beta: {config['loss'].get('allele_beta', 0.9999)}")
        print(f"  normalize_allele_weights: {config['loss']['normalize_allele_weights']}")
    print()
    
    if parallel_folds and n_gpus > 1:
        # Parallel fold training
        print(f"  Training {n_folds} folds in parallel on {n_gpus} GPUs...")
        fold_auprcs = train_folds_parallel(
            config=config,
            trial_number=trial.number,
            output_dir=output_dir,
            max_epochs=max_epochs,
            n_folds=n_folds,
            n_gpus=n_gpus,
        )
        
        # Check for failures
        if None in fold_auprcs:
            print("  Some folds failed!")
            valid_auprcs = [a for a in fold_auprcs if a is not None]
            if not valid_auprcs:
                print("  All folds failed - pruning trial")
                raise optuna.TrialPruned("All folds failed")
            avg_auprc = sum(valid_auprcs) / len(valid_auprcs)
        else:
            avg_auprc = sum(fold_auprcs) / len(fold_auprcs)
    else:
        # Sequential fold training
        fold_auprcs = []
        
        for fold in range(n_folds):
            print(f"  Training fold {fold}...", end=" ", flush=True)

            # Use first GPU for sequential training
            gpu_id = 0

            auprc = train_fold_for_tuning(
                config=config,
                fold=fold,
                trial_number=trial.number,
                output_dir=output_dir,
                max_epochs=max_epochs,
                gpu_id=gpu_id,
            )
            
            if auprc is None:
                print("FAILED - pruning trial")
                raise optuna.TrialPruned(f"Fold {fold} training failed")
            
            fold_auprcs.append(auprc)
            print(f"AUPRC = {auprc:.4f}")
            
            # Report intermediate value for pruning
            trial.report(sum(fold_auprcs) / len(fold_auprcs), fold)
            
            # Handle pruning
            if trial.should_prune():
                print("  Trial pruned!")
                raise optuna.TrialPruned()
        
        avg_auprc = sum(fold_auprcs) / len(fold_auprcs)
    
    print(f"\n  Average AUPRC: {avg_auprc:.4f}")
    
    # Save trial config
    trial_config_path = output_dir / f"trial_{trial.number}" / "config.yaml"
    trial_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trial_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return avg_auprc


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for DinoMHC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Base config file"
    )
    parser.add_argument(
        "--n_trials", type=int, default=100,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds for CV (use fewer for faster tuning)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10,
        help="Max epochs per fold during tuning"
    )
    parser.add_argument(
        "--study_name", type=str, default=None,
        help="Optuna study name (for resuming)"
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/optuna",
        help="Output directory for tuning results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampler"
    )
    parser.add_argument(
        "--n_startup_trials", type=int, default=10,
        help="Number of random trials before TPE"
    )
    parser.add_argument(
        "--pruning", action="store_true",
        help="Enable trial pruning"
    )
    # Parallel training options
    parser.add_argument(
        "--parallel_folds", action="store_true",
        help="Train folds in parallel on multiple GPUs"
    )
    parser.add_argument(
        "--n_gpus", type=int, default=None,
        help="Number of GPUs to use for parallel fold training. "
             "If not specified, auto-detects available GPUs."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel Optuna trials (requires shared storage). "
             "Use this to run multiple trials in parallel across machines."
    )
    
    return parser.parse_args()


def main():
    """Main tuning function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Load base config
    base_config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Study name
    study_name = args.study_name or f"dinomhc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Detect available GPUs
    n_gpus = args.n_gpus
    if n_gpus is None:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print("\n" + "="*60)
    print("DinoMHC Hyperparameter Tuning")
    print("="*60)
    print(f"\nStudy name: {study_name}")
    print(f"Base config: {args.config}")
    print(f"N trials: {args.n_trials}")
    print(f"N folds: {args.n_folds}")
    print(f"Max epochs per fold: {args.max_epochs}")
    print(f"Output directory: {output_dir}")
    print(f"Available GPUs: {n_gpus}")
    print(f"Parallel folds: {args.parallel_folds}")
    if args.parallel_folds and n_gpus > 1:
        print(f"  -> Will train folds in batches of {n_gpus} for balanced GPU utilization")
    print()
    
    # Create sampler
    sampler = TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
    )
    
    # Create pruner (optional) - disable pruning for parallel folds
    if args.parallel_folds and n_gpus > 1:
        pruner = optuna.pruners.NopPruner()
        if args.pruning:
            print("Note: Pruning is disabled when using parallel folds.")
    else:
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,  # Number of folds before pruning
        ) if args.pruning else optuna.pruners.NopPruner()
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="maximize",  # Maximize AUPRC
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial=trial,
            base_config=base_config,
            output_dir=output_dir,
            max_epochs=args.max_epochs,
            n_folds=args.n_folds,
            parallel_folds=args.parallel_folds,
            n_gpus=n_gpus,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,  # Number of parallel trials
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "="*60)
    print("Tuning Complete!")
    print("="*60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best AUPRC: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config = suggest_hyperparameters_from_params(study.best_trial.params, base_config)
    best_config_path = output_dir / "best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    print(f"\nBest config saved to: {best_config_path}")
    
    # Save study statistics
    stats = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
    }
    stats_path = output_dir / "study_stats.yaml"
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"Study stats saved to: {stats_path}")
    
    # Print top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number'])}: AUPRC = {row['value']:.4f}")


def suggest_hyperparameters_from_params(params: Dict, base_config: Dict) -> Dict:
    """Reconstruct config from Optuna params."""
    config = copy.deepcopy(base_config)
    
    # Model
    if 'dim' in params:
        config['model']['dim'] = params['dim']
    if 'num_groove_tokens' in params:
        config['model']['num_groove_tokens'] = params['num_groove_tokens']
    if 'num_fusion_layers' in params:
        config['model']['num_fusion_layers'] = params['num_fusion_layers']
    if 'num_geometry_blocks' in params:
        config['model']['num_geometry_blocks'] = params['num_geometry_blocks']
    if 'num_heads' in params:
        config['model']['num_heads'] = params['num_heads']
    if 'dropout' in params:
        config['model']['dropout'] = params['dropout']
    
    # Training
    if 'learning_rate' in params:
        config['training']['learning_rate'] = params['learning_rate']
    if 'weight_decay' in params:
        config['training']['weight_decay'] = params['weight_decay']
    if 'warmup_steps' in params:
        config['training']['warmup_steps'] = params['warmup_steps']
    if 'encoder_lr_factor' in params:
        config['training']['encoder_lr_factor'] = params['encoder_lr_factor']
    
    # Loss
    if 'use_allele_balanced_loss' in params:
        config['loss']['use_allele_balanced_loss'] = params['use_allele_balanced_loss']
    if 'focal_alpha' in params:
        config['loss']['focal_alpha'] = params['focal_alpha']
    if 'focal_gamma' in params:
        config['loss']['focal_gamma'] = params['focal_gamma']
    if 'allele_weight_type' in params:
        config['loss']['allele_weight_type'] = params['allele_weight_type']
    if 'allele_beta' in params:
        config['loss']['allele_beta'] = params['allele_beta']
    if 'normalize_allele_weights' in params:
        config['loss']['normalize_allele_weights'] = params['normalize_allele_weights']
    if 'pos_weight' in params:
        config['loss']['pos_weight'] = params['pos_weight']
    if 'label_smoothing' in params:
        config['loss']['label_smoothing'] = params['label_smoothing']

    return config


if __name__ == "__main__":
    main()
