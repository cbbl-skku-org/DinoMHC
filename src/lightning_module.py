"""
PyTorch Lightning Module for DinoMHC Training.

This module wraps the DinoMHC model for training with PyTorch Lightning,
including training/validation/test steps, loss functions, metrics, and logging.
Supports per-MHC allele metric tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from torchmetrics import (
    Accuracy, 
    AUROC, 
    AveragePrecision, 
    F1Score,
    Precision,
    Recall,
    MetricCollection,
    PrecisionRecallCurve,
    MatthewsCorrCoef
)

from .model import DinoMHC


def find_optimal_threshold_pr(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal threshold based on Precision-Recall curve.

    Uses F-beta score to balance precision and recall.
    For beta=1, it's the standard F1 score.
    For beta>1, recall is weighted more heavily.
    For beta<1, precision is weighted more heavily.

    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth labels
        beta: Beta value for F-beta score

    Returns:
        optimal_threshold: Threshold that maximizes F-beta score
        best_f_beta: The best F-beta score achieved
    """
    # Compute precision-recall curve
    pr_curve = PrecisionRecallCurve(task='binary')
    precision, recall, thresholds = pr_curve(predictions, targets.long())

    # Compute F-beta score for each threshold
    # F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision[:-1] * recall[:-1]) / (
        beta_sq * precision[:-1] + recall[:-1] + 1e-10
    )

    # Find threshold with maximum F-beta
    best_idx = torch.argmax(f_beta)
    optimal_threshold = thresholds[best_idx].item()
    best_f_beta = f_beta[best_idx].item()

    return optimal_threshold, best_f_beta


def find_optimal_threshold_auprc(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Find optimal threshold that maximizes AUPRC (using precision-recall curve).
    The optimal threshold is chosen as the one that gives the best F1 score.

    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth labels

    Returns:
        optimal_threshold: Threshold that maximizes F1 score on PR curve
    """
    return find_optimal_threshold_pr(predictions, targets, beta=1.0)[0]


def compute_binary_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Ensure we have enough samples
    if len(predictions) < 2:
        return {}

    # Check if we have both classes
    unique_targets = torch.unique(targets)
    if len(unique_targets) < 2:
        # Can't compute AUROC/AUPRC with single class
        return {}

    binary_preds = (predictions >= threshold).long()

    # Basic counts
    tp = ((binary_preds == 1) & (targets == 1)).sum().float()
    fp = ((binary_preds == 1) & (targets == 0)).sum().float()
    fn = ((binary_preds == 0) & (targets == 1)).sum().float()
    tn = ((binary_preds == 0) & (targets == 0)).sum().float()

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # MCC
    mcc_num = (tp * tn - fp * fn)
    mcc_denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10)
    mcc = mcc_num / mcc_denom

    # AUROC and AUPRC
    try:
        auroc_metric = AUROC(task='binary')
        auroc = auroc_metric(predictions, targets.long()).item()
    except:
        auroc = 0.0

    try:
        auprc_metric = AveragePrecision(task='binary')
        auprc = auprc_metric(predictions, targets.long()).item()
    except:
        auprc = 0.0

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'f1': f1.item(),
        'mcc': mcc.item(),
        'auroc': auroc,
        'auprc': auprc,
        'n_samples': len(predictions),
        'n_positive': int(targets.sum().item()),
        'n_negative': int((1 - targets).sum().item()),
    }


def compute_per_mhc_metrics_with_optimal_thresholds(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mhc_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute per-MHC metrics with optimal thresholds based on AUPRC for each MHC.

    Args:
        predictions: All predictions [N]
        targets: All targets [N]
        mhc_names: MHC name for each sample [N]

    Returns:
        per_mhc_metrics: Dict[mhc_name -> metrics_dict]
        per_mhc_optimal_thresholds: Dict[mhc_name -> optimal_threshold]
    """
    # Group by MHC
    mhc_data = defaultdict(lambda: {'preds': [], 'targets': []})
    for pred, target, mhc in zip(predictions.tolist(), targets.tolist(), mhc_names):
        mhc_data[mhc]['preds'].append(pred)
        mhc_data[mhc]['targets'].append(target)

    per_mhc_metrics = {}
    per_mhc_optimal_thresholds = {}

    for mhc_name, data in mhc_data.items():
        preds = torch.tensor(data['preds'])
        tgts = torch.tensor(data['targets'])

        # Skip if not enough samples or only one class
        if len(preds) < 10 or len(torch.unique(tgts)) < 2:
            continue

        # Find optimal threshold for this MHC based on F1 score (AUPRC-based)
        try:
            optimal_thresh = find_optimal_threshold_auprc(preds, tgts)
            per_mhc_optimal_thresholds[mhc_name] = optimal_thresh
        except:
            optimal_thresh = 0.5
            per_mhc_optimal_thresholds[mhc_name] = 0.5

        # Compute metrics at threshold=0.5
        metrics_05 = compute_binary_metrics(preds, tgts, threshold=0.5)

        # Compute metrics at optimal threshold
        metrics_opt = compute_binary_metrics(preds, tgts, threshold=optimal_thresh)

        # Store both
        per_mhc_metrics[mhc_name] = {
            # Threshold-independent metrics (same for both)
            'auroc': metrics_05.get('auroc', 0.0),
            'auprc': metrics_05.get('auprc', 0.0),
            'n_samples': metrics_05.get('n_samples', 0),
            'n_positive': metrics_05.get('n_positive', 0),
            'n_negative': metrics_05.get('n_negative', 0),
            # Metrics at threshold=0.5
            'accuracy_0.5': metrics_05.get('accuracy', 0.0),
            'precision_0.5': metrics_05.get('precision', 0.0),
            'recall_0.5': metrics_05.get('recall', 0.0),
            'f1_0.5': metrics_05.get('f1', 0.0),
            'mcc_0.5': metrics_05.get('mcc', 0.0),
            # Metrics at optimal threshold
            'optimal_threshold': optimal_thresh,
            'accuracy_opt': metrics_opt.get('accuracy', 0.0),
            'precision_opt': metrics_opt.get('precision', 0.0),
            'recall_opt': metrics_opt.get('recall', 0.0),
            'f1_opt': metrics_opt.get('f1', 0.0),
            'mcc_opt': metrics_opt.get('mcc', 0.0),
        }

    return per_mhc_metrics, per_mhc_optimal_thresholds


class DinoMHCLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for DinoMHC.
    
    Supports:
    - Binary classification (presentation/binding prediction)
    - Regression (affinity prediction)
    - Multi-task learning (optional)
    - Gradient clipping and learning rate scheduling
    - Comprehensive metric logging
    - Per-MHC allele metric tracking
    
    Args:
        config: Model configuration dictionary
        learning_rate: Base learning rate
        weight_decay: L2 regularization weight
        warmup_steps: Number of warmup steps for scheduler
        max_steps: Maximum training steps (for scheduler)
        encoder_lr_factor: Learning rate multiplier for encoder layers
        pos_weight: Positive class weight for BCEWithLogitsLoss
        label_smoothing: Label smoothing factor (0 to 1)
        track_per_mhc: Whether to track per-MHC metrics
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        encoder_lr_factor: float = 0.1,
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0,
        track_per_mhc: bool = True,
    ):
        super().__init__()
        
        # Save hyperparameters (excluding config to avoid nested dicts issues)
        self.save_hyperparameters()
        
        # Store config separately
        self.config = config or {}
        print(self.config)
        # Build model
        self.model = DinoMHC(config)
        
        # Task type from config
        self.task_type = self.config.get('task_head', 'presentation')
        
        # Per-MHC tracking
        self.track_per_mhc = track_per_mhc
        
        # Setup loss function based on task
        self._setup_loss(pos_weight, label_smoothing)
        
        # Setup metrics
        self._setup_metrics()
        
    def _setup_loss(
        self, 
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0
    ):
        """Setup loss function based on task type."""
        self.label_smoothing = label_smoothing
        
        if self.task_type in ['presentation', 'classification']:
            # Binary classification loss
            # Note: Model's PresentationHead already applies sigmoid,
            # so we use BCELoss instead of BCEWithLogitsLoss
            if pos_weight is not None:
                # For BCELoss with pos_weight, we need to apply it manually
                self.pos_weight = pos_weight
                self.criterion = nn.BCELoss(reduction='none')
            else:
                self.pos_weight = None
                self.criterion = nn.BCELoss()
                
        elif self.task_type == 'affinity':
            # Regression loss
            self.criterion = nn.MSELoss()
            self.pos_weight = None
            
        elif self.task_type == 'contact':
            # Binary classification for each contact
            self.criterion = nn.BCELoss()
            self.pos_weight = None
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _setup_metrics(self):
        """Setup metrics for training and validation."""
        if self.task_type in ['presentation', 'classification']:
            # Classification metrics
            metrics = MetricCollection({
                'accuracy': Accuracy(task='binary'),
                'auroc': AUROC(task='binary'),
                'auprc': AveragePrecision(task='binary'),
                'f1': F1Score(task='binary'),
                'precision': Precision(task='binary'),
                'recall': Recall(task='binary'),
                'mcc': MatthewsCorrCoef(task='binary'),
            })
            
            # Clone for train/val/test
            self.train_metrics = metrics.clone(prefix='train/')
            self.val_metrics = metrics.clone(prefix='val/')
            self.test_metrics = metrics.clone(prefix='test/')
            
            # Store predictions and targets for threshold optimization
            self.val_predictions = []
            self.val_targets = []
            self.val_mhc_names = []  # Store MHC names for per-MHC metrics
            self.test_predictions = []
            self.test_targets = []
            self.test_mhc_names = []  # Store MHC names for per-MHC metrics
            
            # Optimal threshold (tuned on validation set)
            self.optimal_threshold = 0.5
            
            # Per-MHC optimal thresholds
            self.per_mhc_thresholds: Dict[str, float] = {}
            
        elif self.task_type == 'affinity':
            # Regression metrics (using custom tracking)
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
        
    def forward(
        self,
        peptide_tokens: torch.Tensor,
        mhc_tokens: torch.Tensor,
        peptide_mask: Optional[torch.Tensor] = None,
        mhc_mask: Optional[torch.Tensor] = None,
        nflank_len: Optional[torch.Tensor] = None,
        cflank_len: Optional[torch.Tensor] = None,
        original_peptide_len: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(
            peptide_tokens=peptide_tokens,
            mhc_tokens=mhc_tokens,
            peptide_mask=peptide_mask,
            mhc_mask=mhc_mask,
            nflank_len=nflank_len,
            cflank_len=cflank_len,
            original_peptide_len=original_peptide_len,
            return_attention=return_attention
        )
    
    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with optional label smoothing and class weighting."""
        # Apply label smoothing for classification
        if self.label_smoothing > 0 and self.task_type in ['presentation', 'classification']:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Ensure correct shapes
        predictions = predictions.squeeze(-1)
        targets = targets.float()
        
        # Compute loss in full precision (BCE is not safe with AMP)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            predictions = predictions.float()
            targets = targets.float()
            
            if self.pos_weight is not None:
                # Manual pos_weight application for BCELoss
                loss = self.criterion(predictions, targets)
                weights = torch.where(
                    targets >= 0.5,
                    torch.tensor(self.pos_weight, device=targets.device),
                    torch.tensor(1.0, device=targets.device)
                )
                loss = (loss * weights).mean()
            else:
                loss = self.criterion(predictions, targets)
        
        return loss
    
    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        stage: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test."""
        # Extract batch data
        peptide_tokens = batch['peptide_tokens']
        peptide_mask = batch['peptide_mask']
        mhc_tokens = batch['mhc_tokens']
        mhc_mask = batch['mhc_mask']
        targets = batch['label']

        # Extract flank parameters if present
        nflank_len = batch.get('nflank_len', None)
        cflank_len = batch.get('cflank_len', None)
        original_peptide_len = batch.get('original_peptide_len', None)

        # Forward pass
        outputs = self(
            peptide_tokens=peptide_tokens,
            mhc_tokens=mhc_tokens,
            peptide_mask=peptide_mask,
            mhc_mask=mhc_mask,
            nflank_len=nflank_len,
            cflank_len=cflank_len,
            original_peptide_len=original_peptide_len,
            return_attention=False
        )

        predictions = outputs['prediction']

        # Compute loss
        loss = self._compute_loss(predictions, targets)

        return {
            'loss': loss,
            'predictions': predictions.squeeze(-1),
            'targets': targets.float()
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, batch_idx, stage='train')
        
        # Log loss
        self.log(
            'train/loss', 
            outputs['loss'], 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True,
            batch_size=batch['peptide_tokens'].size(0)
        )
        
        # Update and log metrics
        if self.train_metrics is not None:
            # Detach predictions to avoid gradient issues
            preds = outputs['predictions'].detach()
            targets = outputs['targets'].detach().long()
            
            self.train_metrics.update(preds, targets)
        
        return outputs['loss']
    
    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        if self.train_metrics is not None:
            metrics = self.train_metrics.compute()
            self.log_dict(metrics, on_epoch=True)
            self.train_metrics.reset()
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self._shared_step(batch, batch_idx, stage='val')
        
        # Log loss
        self.log(
            'val/loss', 
            outputs['loss'], 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True,
            batch_size=batch['peptide_tokens'].size(0)
        )
        
        # Update metrics
        if self.val_metrics is not None:
            preds = outputs['predictions'].detach()
            targets = outputs['targets'].detach().long()
            self.val_metrics.update(preds, targets)
            
            # Store for threshold optimization
            self.val_predictions.append(preds.cpu())
            self.val_targets.append(targets.cpu())
            
            # Store MHC names for per-MHC metrics
            if self.track_per_mhc and 'mhc_name' in batch:
                self.val_mhc_names.extend(batch['mhc_name'])
        
        return outputs
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end with macro metrics per MHC."""
        if self.val_metrics is None or len(self.val_predictions) == 0:
            return

        all_preds = torch.cat(self.val_predictions, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)

        # Compute per-MHC metrics with optimal thresholds
        if self.track_per_mhc and len(self.val_mhc_names) == len(all_preds):
            per_mhc_metrics, per_mhc_optimal_thresholds = compute_per_mhc_metrics_with_optimal_thresholds(
                all_preds, all_targets, self.val_mhc_names
            )

            # Store for later use
            self.per_mhc_thresholds = per_mhc_optimal_thresholds
            self._val_per_mhc_metrics = per_mhc_metrics

            # Compute macro averages (average across MHCs)
            if per_mhc_metrics:
                metric_names = ['auroc', 'auprc',
                               'accuracy_0.5', 'precision_0.5', 'recall_0.5', 'f1_0.5', 'mcc_0.5',
                               'accuracy_opt', 'precision_opt', 'recall_opt', 'f1_opt', 'mcc_opt']

                macro_metrics = {}
                for metric_name in metric_names:
                    values = [m[metric_name] for m in per_mhc_metrics.values() if metric_name in m]
                    if values:
                        macro_metrics[f'val/macro_{metric_name}'] = sum(values) / len(values)

                # Log macro metrics
                self.log_dict(macro_metrics, on_epoch=True, prog_bar=True)

                # Log individual MHC metrics
                for mhc_name, metrics in per_mhc_metrics.items():
                    # Sanitize MHC name for logging (replace special characters)
                    safe_mhc_name = mhc_name.replace('*', '_').replace(':', '_')
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.log(f'val/{safe_mhc_name}/{metric_name}', value, on_epoch=True)

                # Log the main metric for model selection (macro AUPRC)
                self.log('val/auprc', macro_metrics.get('val/macro_auprc', 0.0), on_epoch=True, prog_bar=True)

        # Clear stored predictions
        self.val_predictions = []
        self.val_targets = []
        self.val_mhc_names = []
    
    def _compute_per_mhc_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mhc_names: List[str],
        stage: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each MHC allele separately.
        
        Args:
            predictions: All predictions
            targets: All targets
            mhc_names: List of MHC names for each sample
            stage: 'val' or 'test'
        
        Returns:
            Dictionary mapping MHC name to metrics dict
        """
        # Group predictions and targets by MHC
        mhc_data: Dict[str, Dict[str, List]] = defaultdict(lambda: {'preds': [], 'targets': []})
        
        for pred, target, mhc in zip(predictions.tolist(), targets.tolist(), mhc_names):
            mhc_data[mhc]['preds'].append(pred)
            mhc_data[mhc]['targets'].append(target)
        
        # Compute metrics for each MHC
        per_mhc_metrics = {}
        
        for mhc_name, data in mhc_data.items():
            preds = torch.tensor(data['preds'])
            tgts = torch.tensor(data['targets']).long()
            
            metrics = compute_binary_metrics(preds, tgts, threshold=self.optimal_threshold)
            
            if metrics:  # Only include if we could compute metrics
                per_mhc_metrics[mhc_name] = metrics
                
                # Also compute per-MHC optimal threshold
                if len(preds) >= 10:  # Need enough samples
                    try:
                        opt_thresh, _ = find_optimal_threshold_pr(preds, tgts)
                        self.per_mhc_thresholds[mhc_name] = opt_thresh
                    except:
                        pass
        
        return per_mhc_metrics
    
    def _log_per_mhc_summary(
        self,
        per_mhc_metrics: Dict[str, Dict[str, float]],
        stage: str
    ):
        """
        Log summary statistics across all MHCs.
        
        Args:
            per_mhc_metrics: Dictionary of metrics for each MHC
            stage: 'val' or 'test'
        """
        if not per_mhc_metrics:
            return
        
        # Collect metrics across MHCs
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auroc', 'auprc']
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in per_mhc_metrics.values() if metric_name in m]
            
            if values:
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                
                # Log mean, min, max across MHCs
                self.log(f'{stage}/per_mhc_{metric_name}_mean', mean_val, on_epoch=True)
                self.log(f'{stage}/per_mhc_{metric_name}_min', min_val, on_epoch=True)
                self.log(f'{stage}/per_mhc_{metric_name}_max', max_val, on_epoch=True)
        
        # Log number of MHCs evaluated
        self.log(f'{stage}/n_mhc_evaluated', float(len(per_mhc_metrics)), on_epoch=True)
        
        # Store detailed per-MHC metrics as an attribute for later analysis
        if stage == 'val':
            self._val_per_mhc_metrics = per_mhc_metrics
        else:
            self._test_per_mhc_metrics = per_mhc_metrics
    
    def on_test_epoch_start(self):
        self.test_predictions = []
        self.test_targets = []
        self.test_mhc_names = []
    
    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        outputs = self._shared_step(batch, batch_idx, stage='test')
        
        # Log loss
        self.log(
            'test/loss', 
            outputs['loss'], 
            on_step=False, 
            on_epoch=True,
            batch_size=batch['peptide_tokens'].size(0)
        )
        
        # Update metrics
        if self.test_metrics is not None:
            preds = outputs['predictions'].detach()
            targets = outputs['targets'].detach().long()
            self.test_metrics.update(preds, targets)
            
            # Store for threshold-based metrics
            self.test_predictions.append(preds.cpu())
            self.test_targets.append(targets.cpu())
            
            # Store MHC names for per-MHC metrics
            if self.track_per_mhc and 'mhc_name' in batch:
                self.test_mhc_names.extend(batch['mhc_name'])
        
        return outputs
    
    def on_test_epoch_end(self):
        """Log test metrics at epoch end with macro metrics per MHC."""
        if self.test_metrics is None or len(self.test_predictions) == 0:
            return

        all_preds = torch.cat(self.test_predictions, dim=0)
        all_targets = torch.cat(self.test_targets, dim=0)

        # Compute per-MHC metrics with BOTH threshold=0.5 and optimal thresholds
        if self.track_per_mhc and len(self.test_mhc_names) == len(all_preds):
            # Get per-MHC optimal thresholds from validation (or compute new ones for test)
            per_mhc_metrics, per_mhc_optimal_thresholds = compute_per_mhc_metrics_with_optimal_thresholds(
                all_preds, all_targets, self.test_mhc_names
            )

            # Store for later use
            self._test_per_mhc_metrics = per_mhc_metrics

            # Compute macro averages (average across MHCs)
            if per_mhc_metrics:
                metric_names = ['auroc', 'auprc',
                               'accuracy_0.5', 'precision_0.5', 'recall_0.5', 'f1_0.5', 'mcc_0.5',
                               'accuracy_opt', 'precision_opt', 'recall_opt', 'f1_opt', 'mcc_opt']

                macro_metrics = {}
                for metric_name in metric_names:
                    values = [m[metric_name] for m in per_mhc_metrics.values() if metric_name in m]
                    if values:
                        macro_metrics[f'test/macro_{metric_name}'] = sum(values) / len(values)

                # Log macro metrics
                self.log_dict(macro_metrics, on_epoch=True)

                # Log individual MHC metrics
                for mhc_name, metrics in per_mhc_metrics.items():
                    # Sanitize MHC name for logging
                    safe_mhc_name = mhc_name.replace('*', '_').replace(':', '_')
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.log(f'test/{safe_mhc_name}/{metric_name}', value, on_epoch=True)

                # Print summary to console
                print("\n" + "=" * 80)
                print("TEST RESULTS - MACRO METRICS (AVERAGED ACROSS MHCs)")
                print("=" * 80)
                print(f"Number of MHCs evaluated: {len(per_mhc_metrics)}")
                print("\nThreshold-independent metrics:")
                print(f"  Macro AUROC: {macro_metrics.get('test/macro_auroc', 0.0):.4f}")
                print(f"  Macro AUPRC: {macro_metrics.get('test/macro_auprc', 0.0):.4f}")
                print("\nMetrics at threshold=0.5:")
                print(f"  Macro Accuracy: {macro_metrics.get('test/macro_accuracy_0.5', 0.0):.4f}")
                print(f"  Macro Precision: {macro_metrics.get('test/macro_precision_0.5', 0.0):.4f}")
                print(f"  Macro Recall: {macro_metrics.get('test/macro_recall_0.5', 0.0):.4f}")
                print(f"  Macro F1: {macro_metrics.get('test/macro_f1_0.5', 0.0):.4f}")
                print(f"  Macro MCC: {macro_metrics.get('test/macro_mcc_0.5', 0.0):.4f}")
                print("\nMetrics at optimal thresholds (per-MHC):")
                print(f"  Macro Accuracy: {macro_metrics.get('test/macro_accuracy_opt', 0.0):.4f}")
                print(f"  Macro Precision: {macro_metrics.get('test/macro_precision_opt', 0.0):.4f}")
                print(f"  Macro Recall: {macro_metrics.get('test/macro_recall_opt', 0.0):.4f}")
                print(f"  Macro F1: {macro_metrics.get('test/macro_f1_opt', 0.0):.4f}")
                print(f"  Macro MCC: {macro_metrics.get('test/macro_mcc_opt', 0.0):.4f}")
                print("=" * 80 + "\n")

    
    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        # Extract batch data
        peptide_tokens = batch['peptide_tokens']
        peptide_mask = batch['peptide_mask']
        mhc_tokens = batch['mhc_tokens']
        mhc_mask = batch['mhc_mask']

        # Extract flank parameters if present
        nflank_len = batch.get('nflank_len', None)
        cflank_len = batch.get('cflank_len', None)
        original_peptide_len = batch.get('original_peptide_len', None)

        # Forward pass
        outputs = self(
            peptide_tokens=peptide_tokens,
            mhc_tokens=mhc_tokens,
            peptide_mask=peptide_mask,
            mhc_mask=mhc_mask,
            nflank_len=nflank_len,
            cflank_len=cflank_len,
            original_peptide_len=original_peptide_len,
            return_attention=False
        )

        result = {
            'predictions': outputs['prediction'].squeeze(-1),
            'idx': batch.get('idx', torch.arange(len(peptide_tokens)))
        }

        # Include targets if available
        if 'label' in batch:
            result['targets'] = batch['label']

        return result
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Separate encoder and other parameters for different learning rates
        encoder_params = []
        other_params = []
        
        encoder_type = self.config.get('encoder_type', 'embedding')
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Check if this is an encoder parameter
            if 'peptide_encoder' in name or 'mhc_encoder' in name or 'shared_esm_encoder' in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.hparams.learning_rate * self.hparams.encoder_lr_factor,
                'name': 'encoder'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.hparams.learning_rate,
                'name': 'other'
            })
        
        # Initialize optimizer
        
        if self.config['training_strategy'].startswith('deepspeed'):
            try:
                from deepspeed.ops.adam import FusedAdam
                optimizer = FusedAdam(
                    param_groups,
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    adam_w_mode=True
                )
            except ImportError:
                raise ImportError("DeepSpeed FusedAdam optimizer not found. Please install DeepSpeed.")
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Learning rate scheduler with warmup
        def lr_lambda(current_step: int) -> float:
            warmup_steps = self.hparams.warmup_steps
            max_steps = self.hparams.max_steps
            
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay after warmup
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'monitor': 'val/auprc',  # Monitor macro AUPRC for model selection
            }
        }
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about encoder configuration."""
        return self.model.get_encoder_info()
    
    def freeze_encoders(self):
        """Freeze encoder parameters."""
        self.model.freeze_encoders()
    
    def unfreeze_encoder_layers(self, num_layers: int = 2):
        """Unfreeze top N layers of ESM encoders."""
        self.model.unfreeze_encoder_layers(num_layers)
    
    def get_optimal_threshold(self) -> float:
        """Return the optimal threshold tuned on validation set."""
        return self.optimal_threshold
    
    def set_optimal_threshold(self, threshold: float):
        """Manually set the optimal threshold."""
        self.optimal_threshold = threshold
    
    def get_per_mhc_thresholds(self) -> Dict[str, float]:
        """Return optimal thresholds for each MHC allele."""
        return self.per_mhc_thresholds
    
    def get_per_mhc_metrics(self, stage: str = 'val') -> Dict[str, Dict[str, float]]:
        """
        Get detailed per-MHC metrics from the last epoch.
        
        Args:
            stage: 'val' or 'test'
        
        Returns:
            Dictionary mapping MHC name to metrics dict
        """
        if stage == 'val':
            return getattr(self, '_val_per_mhc_metrics', {})
        else:
            return getattr(self, '_test_per_mhc_metrics', {})
    
    def export_per_mhc_metrics_csv(self, filepath: str, stage: str = 'val'):
        """
        Export per-MHC metrics to a CSV file.
        
        Args:
            filepath: Path to save CSV
            stage: 'val' or 'test'
        """
        import csv
        
        per_mhc = self.get_per_mhc_metrics(stage)
        if not per_mhc:
            print(f"No per-MHC metrics available for stage '{stage}'")
            return
        
        # Get all metric names
        metric_names = ['n_samples', 'n_positive', 'n_negative', 
                        'accuracy', 'precision', 'recall', 'f1', 
                        'mcc', 'auroc', 'auprc']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['mhc_name', 'optimal_threshold'] + metric_names
            writer.writerow(header)
            
            # Data rows
            for mhc_name, metrics in sorted(per_mhc.items()):
                threshold = self.per_mhc_thresholds.get(mhc_name, self.optimal_threshold)
                row = [mhc_name, threshold]
                row.extend([metrics.get(m, '') for m in metric_names])
                writer.writerow(row)
        
        print(f"Per-MHC metrics saved to: {filepath}")


class AlleleBalancedFocalLoss(nn.Module):
    """
    Focal Loss with per-allele balancing to handle both class and allele imbalance.

    This loss addresses two types of imbalance:
    1. Class imbalance: negative vs positive samples (via focal loss)
    2. Allele imbalance: some MHC alleles have many samples, others have few

    The per-allele weights are computed using inverse frequency:
        weight_allele = N_total / (N_alleles * N_samples_for_allele)

    Or using effective number of samples (Class-Balanced Loss):
        weight_allele = (1 - beta) / (1 - beta^N_samples_for_allele)

    Args:
        alpha: Weighting factor for positive class (focal loss)
        gamma: Focusing parameter (focal loss)
        allele_weight_type: How to compute allele weights ('inverse_freq' or 'effective_num')
        beta: Beta parameter for effective number weighting (default: 0.9999)
        normalize_weights: Whether to normalize allele weights to have mean=1
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        allele_weight_type: str = 'inverse_freq',  # 'inverse_freq' or 'effective_num'
        beta: float = 0.9999,
        normalize_weights: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.allele_weight_type = allele_weight_type
        self.beta = beta
        self.normalize_weights = normalize_weights

        # Allele statistics (will be updated during training)
        self.allele_counts: Dict[str, int] = {}
        self.allele_weights: Dict[str, float] = {}
        self.total_samples = 0

    def update_allele_statistics(self, allele_counts: Dict[str, int]):
        """
        Update allele statistics and recompute weights.

        Args:
            allele_counts: Dictionary mapping allele name to sample count
        """
        self.allele_counts = allele_counts
        self.total_samples = sum(allele_counts.values())

        # Compute allele weights
        if self.allele_weight_type == 'inverse_freq':
            # Inverse frequency weighting
            n_alleles = len(allele_counts)
            for allele, count in allele_counts.items():
                self.allele_weights[allele] = self.total_samples / (n_alleles * count)

        elif self.allele_weight_type == 'effective_num':
            # Effective number of samples (Class-Balanced Loss)
            for allele, count in allele_counts.items():
                effective_num = (1.0 - self.beta) / (1.0 - self.beta ** count)
                self.allele_weights[allele] = 1.0 / effective_num

        else:
            raise ValueError(f"Unknown allele_weight_type: {self.allele_weight_type}")

        # Normalize weights to have mean = 1 (optional)
        if self.normalize_weights and self.allele_weights:
            mean_weight = sum(self.allele_weights.values()) / len(self.allele_weights)
            self.allele_weights = {
                allele: weight / mean_weight
                for allele, weight in self.allele_weights.items()
            }

    def get_sample_weights(self, mhc_names: List[str], device: torch.device) -> torch.Tensor:
        """
        Get per-sample weights based on allele names.

        Args:
            mhc_names: List of MHC allele names for each sample in batch
            device: Device to place the weights on

        Returns:
            weights: Tensor of shape [batch_size] with per-sample weights
        """
        if not self.allele_weights:
            # No statistics yet, return uniform weights
            return torch.ones(len(mhc_names), device=device)

        weights = []
        for mhc in mhc_names:
            # Use weight if available, otherwise use mean weight (for unseen alleles)
            weight = self.allele_weights.get(mhc, 1.0)
            weights.append(weight)

        return torch.tensor(weights, device=device, dtype=torch.float32)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mhc_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (after sigmoid), shape [batch]
            targets: Binary labels, shape [batch]
            mhc_names: Optional list of MHC allele names for per-allele weighting
        """
        # Compute in full precision
        with torch.amp.autocast(device_type='cuda', enabled=False):
            p = inputs.float()
            targets = targets.float()
            ce_loss = F.binary_cross_entropy(p, targets, reduction='none')

            # Focal loss weighting
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

            loss = focal_weight * ce_loss

            # Apply per-allele weighting if MHC names provided
            if mhc_names is not None and self.allele_weights:
                allele_weights = self.get_sample_weights(mhc_names, inputs.device)
                loss = loss * allele_weights

            return loss.mean()

class DinoMHCWithAlleleBalancedFocalLoss(DinoMHCLightningModule):
    """
    DinoMHC Lightning Module with Allele-Balanced Focal Loss.

    This module handles TWO types of imbalance:
    1. Class imbalance: negative vs positive samples (via focal loss)
    2. Allele imbalance: some MHC alleles have many samples, others have few

    The loss function automatically computes per-allele weights based on
    training set statistics and applies them during training.

    Args:
        focal_alpha: Weighting factor for positive class (focal loss)
        focal_gamma: Focusing parameter (focal loss)
        allele_weight_type: How to compute allele weights ('inverse_freq' or 'effective_num')
        allele_beta: Beta parameter for effective number weighting (only used if allele_weight_type='effective_num')
        normalize_allele_weights: Whether to normalize allele weights to have mean=1
        **kwargs: Other arguments passed to DinoMHCLightningModule
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        allele_weight_type: str = 'inverse_freq',
        allele_beta: float = 0.9999,
        normalize_allele_weights: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Replace criterion with Allele-Balanced Focal Loss for classification tasks
        if self.task_type in ['presentation', 'classification']:
            self.criterion = AlleleBalancedFocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                allele_weight_type=allele_weight_type,
                beta=allele_beta,
                normalize_weights=normalize_allele_weights
            )
            self.pos_weight = None  # Focal loss handles weighting internally

        self.save_hyperparameters('focal_alpha', 'focal_gamma', 'allele_weight_type',
                                   'allele_beta', 'normalize_allele_weights')

        # Flag to track if allele statistics have been computed
        self._allele_stats_computed = False

    def on_train_start(self):
        """Compute allele statistics from training dataset before training starts."""
        super().on_train_start()

        if not self._allele_stats_computed and hasattr(self, 'trainer'):
            # Get training dataloader
            train_loader = self.trainer.train_dataloader

            if train_loader is not None:
                # Count samples per allele
                allele_counts = defaultdict(int)

                # Iterate through dataset (not dataloader to avoid shuffling issues)
                if hasattr(train_loader, 'dataset'):
                    dataset = train_loader.dataset
                    if hasattr(dataset, 'df') and hasattr(dataset, 'mhc_column'):
                        # Direct access to dataframe
                        for allele in dataset.df[dataset.mhc_column]:
                            allele_counts[allele] += 1
                    else:
                        # Fallback: iterate through dataset
                        for i in range(len(dataset)):
                            sample = dataset[i]
                            if 'mhc_name' in sample:
                                allele_counts[sample['mhc_name']] += 1

                # Update loss function with statistics
                if allele_counts and isinstance(self.criterion, AlleleBalancedFocalLoss):
                    self.criterion.update_allele_statistics(dict(allele_counts))
                    self._allele_stats_computed = True

                    # Log statistics
                    n_alleles = len(allele_counts)
                    total_samples = sum(allele_counts.values())
                    min_count = min(allele_counts.values())
                    max_count = max(allele_counts.values())
                    mean_count = total_samples / n_alleles

                    print(f"\n{'='*60}")
                    print(f"Allele Statistics:")
                    print(f"  Total alleles: {n_alleles}")
                    print(f"  Total samples: {total_samples}")
                    print(f"  Samples per allele - Min: {min_count}, Max: {max_count}, Mean: {mean_count:.1f}")
                    print(f"  Imbalance ratio: {max_count / min_count:.2f}x")
                    print(f"\nAllele weights (top 10 by weight):")
                    sorted_weights = sorted(self.criterion.allele_weights.items(),
                                           key=lambda x: x[1], reverse=True)
                    for allele, weight in sorted_weights[:10]:
                        count = allele_counts[allele]
                        print(f"  {allele}: weight={weight:.3f} (n={count})")
                    print(f"{'='*60}\n")

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mhc_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Compute allele-balanced focal loss."""
        predictions = predictions.squeeze(-1)
        targets = targets.float()

        # Pass MHC names to loss function for per-allele weighting
        if isinstance(self.criterion, AlleleBalancedFocalLoss):
            return self.criterion(predictions, targets, mhc_names=mhc_names)
        else:
            return self.criterion(predictions, targets)

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        stage: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test - modified to pass MHC names to loss."""
        # Extract batch data
        peptide_tokens = batch['peptide_tokens']
        peptide_mask = batch['peptide_mask']
        mhc_tokens = batch['mhc_tokens']
        mhc_mask = batch['mhc_mask']
        targets = batch['label']
        mhc_names = batch.get('mhc_name', None)  # Get MHC names

        # Extract flank parameters if present
        nflank_len = batch.get('nflank_len', None)
        cflank_len = batch.get('cflank_len', None)
        original_peptide_len = batch.get('original_peptide_len', None)

        # Forward pass
        outputs = self(
            peptide_tokens=peptide_tokens,
            mhc_tokens=mhc_tokens,
            peptide_mask=peptide_mask,
            mhc_mask=mhc_mask,
            nflank_len=nflank_len,
            cflank_len=cflank_len,
            original_peptide_len=original_peptide_len,
            return_attention=False
        )

        predictions = outputs['prediction']

        # Compute loss (pass MHC names for allele balancing)
        loss = self._compute_loss(predictions, targets, mhc_names=mhc_names)

        return {
            'loss': loss,
            'predictions': predictions.squeeze(-1),
            'targets': targets.float()
        }


def create_lightning_module(
    config: Optional[Dict[str, Any]] = None,
    use_allele_balanced_loss: bool = False,
    **kwargs
) -> DinoMHCLightningModule:
    """
    Factory function to create appropriate lightning module.

    Args:
        config: Model configuration
        use_allele_balanced_loss: Whether to use Allele-Balanced Focal Loss
                                   (for both class AND allele imbalanced data)
        use_contrastive_loss: Whether to use Hybrid Focal + Contrastive Loss
                              (recommended for handling sequence-similar alleles
                               with different binding specificities)
        **kwargs: Additional arguments passed to module

    Returns:
        DinoMHCLightningModule instance

    Note:
        - If use_contrastive_loss=True: Uses HybridFocalContrastiveLoss
          which combines focal loss with dual-encoder contrastive learning
          to handle cross-reactive binding and allele-specific patterns
        - If use_allele_balanced_loss=True: Uses AlleleBalancedFocalLoss
          which handles both class imbalance and allele imbalance
        - If both are False: Uses standard BCE loss with optional pos_weight

    Priority: use_contrastive_loss > use_allele_balanced_loss > standard loss
    """
    if use_allele_balanced_loss:
        return DinoMHCWithAlleleBalancedFocalLoss(config=config, **kwargs)
    else:
        return DinoMHCLightningModule(config=config, **kwargs)
