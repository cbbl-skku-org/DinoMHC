#!/usr/bin/env python
"""
Dummy Training Script for DinoMHC.

This script demonstrates:
1. Data loading with proper batching and padding
2. Training loop with mask handling
3. Validation and metrics
4. Learning rate scheduling
5. Checkpointing
6. Support for both embedding and ESM-2 encoders

Can be used as a template for actual training.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Optional
import random
import argparse

sys.path.insert(0, '/home/duongtt/Workdir/MHC/digermhc/src')
from model import DinoMHC

# Check if ESM is available
try:
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False


# =============================================================================
# AMINO ACID UTILITIES
# =============================================================================

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for padding
IDX_TO_AA = {i+1: aa for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS) + 1  # +1 for padding


def encode_sequence(seq: str) -> List[int]:
    """Convert amino acid sequence to token indices (for embedding encoder)."""
    return [AA_TO_IDX.get(aa, 0) for aa in seq]


def decode_sequence(tokens: List[int]) -> str:
    """Convert token indices back to amino acid sequence."""
    return ''.join(IDX_TO_AA.get(t, 'X') for t in tokens if t > 0)


# =============================================================================
# DUMMY DATASET
# =============================================================================

class DummypHLADataset(Dataset):
    """
    Dummy dataset for testing.
    
    Generates random peptide-MHC pairs with synthetic labels.
    In practice, replace with actual data loading.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        min_peptide_len: int = 8,
        max_peptide_len: int = 12,
        mhc_len: int = 180,
        seed: int = 42
    ):
        super().__init__()
        self.num_samples = num_samples
        self.min_peptide_len = min_peptide_len
        self.max_peptide_len = max_peptide_len
        self.mhc_len = mhc_len
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate random data
        self.peptides = []
        self.peptide_lengths = []
        self.mhcs = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random peptide
            pep_len = np.random.randint(min_peptide_len, max_peptide_len + 1)
            peptide = ''.join(random.choices(AMINO_ACIDS, k=pep_len))
            self.peptides.append(peptide)
            self.peptide_lengths.append(pep_len)
            
            # Random MHC
            mhc = ''.join(random.choices(AMINO_ACIDS, k=mhc_len))
            self.mhcs.append(mhc)
            
            # Random label (binary binding)
            # In real data, this would be experimental binding data
            label = float(np.random.random() > 0.5)
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'peptide': self.peptides[idx],
            'peptide_len': self.peptide_lengths[idx],
            'mhc': self.mhcs[idx],
            'label': self.labels[idx]
        }


def create_collate_fn(encoder_type: str = 'embedding'):
    """
    Create a collate function based on encoder type.
    
    For 'embedding': uses simple token indices
    For 'esm2' or 'esm2_shared': uses ESM-2 tokenization
    """
    if encoder_type in ['esm2', 'esm2_shared']:
        if not HAS_ESM:
            raise ImportError("ESM package required for esm2 encoder. Install with: pip install fair-esm")
        
        # Load ESM alphabet for tokenization
        _, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')
        batch_converter = alphabet.get_batch_converter()
        padding_idx = alphabet.padding_idx
        
        def esm_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            """Collate function for ESM-2 encoder."""
            peptides = [item['peptide'] for item in batch]
            mhcs = [item['mhc'] for item in batch]
            labels = [item['label'] for item in batch]
            
            batch_size = len(batch)
            
            # Tokenize peptides using ESM batch converter
            pep_data = [(f"pep_{i}", seq) for i, seq in enumerate(peptides)]
            _, _, peptide_tokens = batch_converter(pep_data)
            
            # Tokenize MHCs using ESM batch converter
            mhc_data = [(f"mhc_{i}", seq) for i, seq in enumerate(mhcs)]
            _, _, mhc_tokens = batch_converter(mhc_data)
            
            # Create masks based on padding token
            peptide_mask = peptide_tokens != padding_idx
            mhc_mask = mhc_tokens != padding_idx
            
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return {
                'peptide_tokens': peptide_tokens,
                'peptide_mask': peptide_mask,
                'mhc_tokens': mhc_tokens,
                'mhc_mask': mhc_mask,
                'labels': labels
            }
        
        return esm_collate_fn
    
    else:
        # Simple embedding collate function
        def embedding_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            """Collate function for embedding encoder."""
            peptides = [item['peptide'] for item in batch]
            mhcs = [item['mhc'] for item in batch]
            labels = [item['label'] for item in batch]
            peptide_lens = [item['peptide_len'] for item in batch]
            
            # Find max peptide length in batch
            max_pep_len = max(peptide_lens)
            mhc_len = len(mhcs[0])  # All MHCs same length
            batch_size = len(batch)
            
            # Pad and encode peptides
            peptide_tokens = torch.zeros(batch_size, max_pep_len, dtype=torch.long)
            peptide_mask = torch.zeros(batch_size, max_pep_len, dtype=torch.bool)
            
            for i, (pep, pep_len) in enumerate(zip(peptides, peptide_lens)):
                tokens = encode_sequence(pep)
                peptide_tokens[i, :pep_len] = torch.tensor(tokens)
                peptide_mask[i, :pep_len] = True
            
            # Encode MHCs (all same length, no padding needed)
            mhc_tokens = torch.zeros(batch_size, mhc_len, dtype=torch.long)
            mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool)
            
            for i, mhc in enumerate(mhcs):
                mhc_tokens[i] = torch.tensor(encode_sequence(mhc))
            
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return {
                'peptide_tokens': peptide_tokens,
                'peptide_mask': peptide_mask,
                'mhc_tokens': mhc_tokens,
                'mhc_mask': mhc_mask,
                'labels': labels
            }
        
        return embedding_collate_fn


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class Trainer:
    """
    Training class for DinoMHC.
    """
    def __init__(
        self,
        model: DinoMHC,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cpu',
        log_interval: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_interval = log_interval
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function (BCE for binary classification)
        self.criterion = nn.BCELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            peptide_tokens = batch['peptide_tokens'].to(self.device)
            peptide_mask = batch['peptide_mask'].to(self.device)
            mhc_tokens = batch['mhc_tokens'].to(self.device)
            mhc_mask = batch['mhc_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                peptide_tokens, mhc_tokens,
                peptide_mask=peptide_mask,
                mhc_mask=mhc_mask
            )
            predictions = outputs['prediction'].squeeze(-1)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * len(labels)
            pred_binary = (predictions > 0.5).float()
            correct += (pred_binary == labels).sum().item()
            total += len(labels)
            
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100 * correct / total:.2f}%")
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / total
        accuracy = correct / total
        
        print(f"  Epoch completed in {epoch_time:.1f}s")
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Run validation."""
        if self.val_loader is None:
            return 0, 0
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            peptide_tokens = batch['peptide_tokens'].to(self.device)
            peptide_mask = batch['peptide_mask'].to(self.device)
            mhc_tokens = batch['mhc_tokens'].to(self.device)
            mhc_mask = batch['mhc_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                peptide_tokens, mhc_tokens,
                peptide_mask=peptide_mask,
                mhc_mask=mhc_mask
            )
            predictions = outputs['prediction'].squeeze(-1)
            
            loss = self.criterion(predictions, labels)
            
            total_loss += loss.item() * len(labels)
            pred_binary = (predictions > 0.5).float()
            correct += (pred_binary == labels).sum().item()
            total += len(labels)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            print(f"\n  Train Loss: {train_loss:.4f}, Train Acc: {100 * train_acc:.2f}%")
            if self.val_loader:
                print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {100 * val_acc:.2f}%")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  ✓ New best validation loss!")
                self.save_checkpoint('best_model.pt')
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.model.config
        }
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training curves."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1 = axes[0]
        ax1.plot(history['train_loss'], label='Train', color='blue')
        if history['val_loss'][0] > 0:
            ax1.plot(history['val_loss'], label='Validation', color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2 = axes[1]
        ax2.plot([100*a for a in history['train_acc']], label='Train', color='blue')
        if history['val_acc'][0] > 0:
            ax2.plot([100*a for a in history['val_acc']], label='Validation', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("matplotlib not available for plotting")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='DinoMHC Training')
    parser.add_argument('--encoder', type=str, default='embedding',
                        choices=['embedding', 'esm2', 'esm2_shared'],
                        help='Encoder type to use')
    parser.add_argument('--unfreeze-layers', type=int, default=2,
                        help='Number of ESM-2 layers to unfreeze (0=freeze all, -1=unfreeze all)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--train-samples', type=int, default=200,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=50,
                        help='Number of validation samples')
    parser.add_argument('--mhc-len', type=int, default=100,
                        help='MHC sequence length (shorter for faster demo)')
    args = parser.parse_args()
    
    # Configuration based on encoder type
    if args.encoder in ['esm2', 'esm2_shared']:
        if not HAS_ESM:
            raise ImportError("ESM package required. Install with: pip install fair-esm")
        config = {
            'dim': 320,  # ESM-2 8M output dim
            'encoder_type': args.encoder,
            'esm_model_name': 'esm2_t6_8M_UR50D',
            'esm_unfreeze_layers': args.unfreeze_layers,
            'esm_unfreeze_embeddings': False,
            'num_groove_tokens': 50,
            'num_fusion_layers': 2,
            'num_geometry_blocks': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'max_peptide_length': 15,
            'task_head': 'presentation'
        }
    else:
        config = {
            'dim': 128,  # Smaller for faster training demo
            'encoder_type': 'embedding',
            'num_groove_tokens': 50,
            'num_fusion_layers': 2,
            'num_geometry_blocks': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'max_peptide_length': 15,
            'task_head': 'presentation'
        }
    
    print("=" * 70)
    print("DINOMHC TRAINING")
    print("=" * 70)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\nTraining settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Val samples: {args.val_samples}")
    print(f"  MHC length: {args.mhc_len}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DummypHLADataset(
        num_samples=args.train_samples, 
        mhc_len=args.mhc_len,
        seed=42
    )
    val_dataset = DummypHLADataset(
        num_samples=args.val_samples,
        mhc_len=args.mhc_len,
        seed=123
    )
    
    # Create collate function based on encoder type
    collate_fn = create_collate_fn(config['encoder_type'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("\nCreating model...")
    model = DinoMHC(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Print encoder info
    encoder_info = model.get_encoder_info()
    print(f"  Encoder type: {encoder_info['encoder_type']}")
    print(f"  Shared encoder: {encoder_info['shared_encoder']}")
    if 'encoder_params' in encoder_info:
        print(f"  Encoder params: {encoder_info['encoder_params']}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device,
        log_interval=5
    )
    
    # Train
    history = trainer.train(args.epochs)
    
    # Plot
    plot_training_history(history, save_path='training_history.png')
    
    # Test inference
    print("\n" + "=" * 70)
    print("TESTING INFERENCE")
    print("=" * 70)
    
    model.eval()
    sample_batch = next(iter(val_loader))
    
    with torch.no_grad():
        outputs = model(
            sample_batch['peptide_tokens'].to(device),
            sample_batch['mhc_tokens'].to(device),
            peptide_mask=sample_batch['peptide_mask'].to(device),
            mhc_mask=sample_batch['mhc_mask'].to(device),
            return_attention=True
        )
    
    print(f"\nSample predictions (first 5):")
    predictions = outputs['prediction'].squeeze().cpu()
    labels = sample_batch['labels']
    for i in range(min(5, len(predictions.shape) and predictions.numel() or 1)):
        pred_val = predictions[i].item() if predictions.numel() > 1 else predictions.item()
        label_val = labels[i].item() if labels.numel() > 1 else labels.item()
        print(f"  Sample {i+1}: Pred={pred_val:.4f}, Label={label_val:.0f}")
    
    print("\nAttention info keys:", list(outputs['attention_info'].keys()))
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
