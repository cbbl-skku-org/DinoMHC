#!/usr/bin/env python
"""
Test script for DinoMHC on benchmark datasets.

This script:
1. Loads a trained model checkpoint
2. Runs inference on benchmark test datasets
3. Computes and reports comprehensive metrics including:
   - Overall metrics (AUROC, AUPRC, Accuracy, F1, MCC, etc.)
   - Per-MHC metrics with optimal thresholds
   - Macro-averaged metrics across MHCs
4. Saves predictions and results to CSV files

Usage:
    # Test on a single benchmark dataset
    python test.py --checkpoint outputs/experiment/fold_0/checkpoints/best.ckpt \
                   --test_file datasets/el/EL_Test_Multiallelic.csv \
                   --output_dir test_results

    # Test on multiple benchmark datasets
    python test.py --checkpoint outputs/experiment/fold_0/checkpoints/best.ckpt \
                   --test_files datasets/el/EL_Test_Multiallelic.csv \
                               datasets/el/IM_Test_EBV.csv \
                               datasets/el/IM_Test_HIV_Acute.csv \
                   --output_dir test_results

    # Test with config file
    python test.py --checkpoint outputs/experiment/fold_0/checkpoints/best.ckpt \
                   --config configs/default.yaml \
                   --test_file datasets/el/EL_Test_Multiallelic.csv
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import RichProgressBar
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import RichProgressBar

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
        'task_head': config['model']['task_head'],
        'use_flanks': config['data']['use_flanks'],
        'flank_pooling': config['model']['flank_pooling'],
        'encoder_type': config['model']['encoder_type'],
        'esm_model_name': config['esm']['model_name'],
        'esm_unfreeze_layers': config['esm']['unfreeze_layers'],
        'esm_unfreeze_embeddings': config['esm']['unfreeze_embeddings'],
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test DinoMHC on Benchmark Datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.ckpt file)"
    )

    # Test data - either single file or multiple files
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument(
        "--test_file", type=str,
        help="Path to a single test CSV file"
    )
    test_group.add_argument(
        "--test_files", type=str, nargs='+',
        help="Paths to multiple test CSV files"
    )

    # Optional config file (can infer from checkpoint if not provided)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file (will try to infer from checkpoint dir if not provided)"
    )

    # Output directory
    parser.add_argument(
        "--output_dir", type=str, default="test_results",
        help="Directory to save test results and predictions"
    )

    # Testing options
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use (only valid when device=cuda)"
    )
    parser.add_argument(
        "--strategy", type=str, default="auto",
        choices=["auto", "dp", "ddp"],
        help="Multi-GPU strategy: 'dp' (DataParallel), 'ddp' (DistributedDataParallel), 'auto' (auto-select)"
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save predictions to CSV file"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def infer_config_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Try to find config.yaml in the checkpoint directory."""
    ckpt_path = Path(checkpoint_path)

    # Try several common locations relative to checkpoint
    possible_config_paths = [
        ckpt_path.parent.parent / "config.yaml",  # outputs/exp/fold_X/config.yaml
        ckpt_path.parent.parent.parent / "config.yaml",  # outputs/exp/config.yaml
    ]
    
    print(possible_config_paths)

    for config_path in possible_config_paths:
        if config_path.exists():
            return str(config_path)

    return None


def get_test_files(args) -> List[str]:
    """Get list of test files from args."""
    if args.test_file:
        return [args.test_file]
    else:
        return args.test_files


def test_benchmark(
    checkpoint_path: str,
    test_files: List[str],
    config: Dict[str, Any],
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    num_gpus: int = 1,
    strategy: str = "auto",
    save_predictions: bool = True,
    verbose: bool = True
):
    """
    Test a model on benchmark datasets.

    Args:
        checkpoint_path: Path to model checkpoint
        test_files: List of paths to test CSV files
        config: Model configuration dict
        output_dir: Directory to save results
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        device: Device to use ('cuda' or 'cpu')
        num_gpus: Number of GPUs to use
        strategy: Multi-GPU strategy ('auto', 'dp', 'ddp')
        save_predictions: Whether to save predictions to CSV
        verbose: Enable verbose output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    if verbose:
        print(f"\n{'='*60}")
        print("DinoMHC Benchmark Testing")
        print(f"{'='*60}\n")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Test files: {len(test_files)}")
        for tf in test_files:
            print(f"  - {tf}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {device}")
        if device == "cuda" and num_gpus > 1:
            print(f"GPUs: {num_gpus} (strategy: {strategy})")
        print()

    # Determine tokenizer type from encoder
    encoder_type = config['model']['encoder_type']
    tokenizer_type = 'esm2' if encoder_type in ['esm2', 'esm2_shared'] else 'embedding'

    # Create data module with test files
    print('====HELLO', config['data']['use_flanks'])
    data_module = MHCPeptideDataModule(
        data_dir=".",  # Not used for test-only
        fold=0,  # Not used for test-only
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_type=tokenizer_type,
        esm_model_name=config['esm']['model_name'],
        max_peptide_length=config['data']['max_peptide_length'],
        max_mhc_length=config['data']['max_mhc_length'],
        use_flanks=config['data']['use_flanks'],
        flank_length=config['data']['flank_length'],
        binarize_labels=True,
        label_threshold=0.5,
        test_files=test_files,
        pin_memory=device == 'cuda'
    )

    # Setup test datasets
    data_module.setup('test')

    if verbose:
        print("Test Dataset Info:")
        print("-" * 40)
        for name, dataset in data_module.test_datasets.items():
            print(f"  {name}: {len(dataset):,} samples")
        print()

    # Load model from checkpoint
    if verbose:
        print("Loading model from checkpoint...")

    # Determine which module class to use based on checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    
    
    model = create_lightning_module(
        config=config['model'],
        use_allele_balanced_loss=config['loss'].get('use_allele_balanced_loss', False),
    )
    
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)

    if verbose:
        print(f"Model loaded successfully!")
        print(f"  Encoder: {encoder_type}")
        print(f"  Task: {model.task_type}")
        print(f"  Use flanks: {config['data']['use_flanks']}")
        print()

    # Create trainer for testing
    callbacks = []
    if verbose:
        try:
            callbacks.append(RichProgressBar())
        except:
            pass

    # Configure multi-GPU settings
    if device == "cuda" and num_gpus > 1:
        devices = num_gpus
        # Auto-select best strategy if not specified
        if strategy == "auto":
            # DDP is generally better for most cases
            strategy = "ddp"
    else:
        devices = 1
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        enable_progress_bar=verbose,
        enable_model_summary=False,
        logger=False,
    )

    # Run testing
    if verbose:
        print("Running inference on test datasets...")
        print()

    test_results = trainer.test(model, data_module)
    
    # Process and save results
    if verbose:
        print("\n" + "="*60)
        print("Test Results Summary")
        print("="*60 + "\n")

    all_results = {}

    # Extract metrics from test results
    for i, (dataset_name, dataset) in enumerate(data_module.test_datasets.items()):
        if verbose:
            print(f"Dataset: {dataset_name}")
            print("-" * 40)

        # Get per-MHC metrics if available
        per_mhc_metrics = getattr(model, '_test_per_mhc_metrics', {})

        # Create results dictionary for this dataset
        result_dict = {
            'dataset': dataset_name,
            'n_samples': len(dataset),
        }

        # Add test metrics from trainer output
        if test_results and i < len(test_results):
            test_result = test_results[i] if isinstance(test_results, list) else test_results
            for key, value in test_result.items():
                if key.startswith('test/'):
                    metric_name = key.replace('test/', '')
                    result_dict[metric_name] = value
                    if verbose and not key.startswith('test/macro_'):
                        print(f"  {metric_name}: {value:.4f}")

        # Print macro-averaged per-MHC metrics
        if verbose and per_mhc_metrics:
            print(f"\n  Per-MHC Metrics (macro-averaged across {len(per_mhc_metrics)} alleles):")
            metric_names = ['auroc', 'auprc', 
                            'accuracy_0.5', 'f1_0.5', 'mcc_0.5', 
                            'accuracy_opt', 'f1_opt', 'mcc_opt']
            for metric_name in metric_names:
                values = [m[metric_name] for m in per_mhc_metrics.values() if metric_name in m]
                if values:
                    macro_value = sum(values) / len(values)
                    print(f"    macro_{metric_name}: {macro_value:.4f}")

        all_results[dataset_name] = result_dict

        # Save per-MHC detailed results
        if per_mhc_metrics and save_predictions:
            per_mhc_df = pd.DataFrame.from_dict(per_mhc_metrics, orient='index')
            per_mhc_csv_path = output_path / f"{dataset_name}_per_mhc_metrics.csv"
            per_mhc_df.to_csv(per_mhc_csv_path, index_label='mhc_allele')
            if verbose:
                print(f"\n  Per-MHC metrics saved to: {per_mhc_csv_path}")

        if verbose:
            print()

    # Save aggregated results
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_csv_path = output_path / "test_results_summary.csv"
    results_df.to_csv(results_csv_path, index_label='dataset')

    if verbose:
        print(f"Summary results saved to: {results_csv_path}")

    # Save predictions if requested
    if save_predictions and hasattr(model, 'test_predictions') and len(model.test_predictions) > 0:
        all_preds = torch.cat(model.test_predictions, dim=0).float().numpy()
        all_targets = torch.cat(model.test_targets, dim=0).float().numpy()

        # Match predictions with original data
        for i, (dataset_name, dataset) in enumerate(data_module.test_datasets.items()):
            # Get original dataframe
            df = dataset.df.copy()
            df = df.iloc[:len(all_preds)]  # Ensure matching length

            # Add predictions (assuming single dataset for now)
            if len(data_module.test_datasets) == 1:
                df['score'] = all_preds

                # Save predictions
                pred_csv_path = output_path / f"{dataset_name}_predictions.csv"
                df.to_csv(pred_csv_path, index=False)

                if verbose:
                    print(f"Predictions saved to: {pred_csv_path}")

    if verbose:
        print("\nTesting complete!")

    return all_results


def main():
    """Main testing function."""
    args = parse_args()

    # Get test files
    test_files = get_test_files(args)

    # Verify test files exist
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"Error: Test file not found: {test_file}")
            sys.exit(1)

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load or infer config
    config_path = args.config
    if config_path is None:
        if args.verbose:
            print("Config not provided, trying to infer from checkpoint directory...")
        config_path = infer_config_from_checkpoint(args.checkpoint)
        if config_path is None:
            print("Error: Could not find config.yaml. Please provide --config explicitly.")
            sys.exit(1)
        if args.verbose:
            print(f"Using config: {config_path}")

    config = load_config(config_path)

    # Run testing
    results = test_benchmark(
        checkpoint_path=args.checkpoint,
        test_files=test_files,
        config=config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        num_gpus=args.num_gpus,
        strategy=args.strategy,
        save_predictions=args.save_predictions,
        verbose=args.verbose
    )

    return results


if __name__ == "__main__":
    main()
