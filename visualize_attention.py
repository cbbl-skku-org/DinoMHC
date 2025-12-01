#!/usr/bin/env python
"""
Attention Visualization for DinoMHC.

This script provides comprehensive visualization of all attention mechanisms
in the model, helping to understand:
1. Which MHC regions are identified as relevant (Region Attention)
2. Which groove positions are extracted (Groove Attention)
3. Peptide-Groove interaction patterns (Fusion Attention)
4. Position importance in peptides (Anchor Positions)

Supports both embedding and ESM-2 encoders.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
import sys
from typing import Dict, Optional, List
import argparse

sys.path.insert(0, '/home/duongtt/Workdir/MHC/digermhc/src')
from model import DinoMHC

# Check for ESM
try:
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False


def get_attention_maps(
    model: DinoMHC,
    peptide_tokens: torch.Tensor,
    mhc_tokens: torch.Tensor,
    peptide_mask: Optional[torch.Tensor] = None,
    mhc_mask: Optional[torch.Tensor] = None
) -> Dict:
    """
    Extract all attention maps from model forward pass.
    
    Returns dict with:
        - groove_extraction: {region_attention, groove_attention, region_importance}
        - position_importance: [batch, pep_len]
        - fusion: {layer_0: {peptide_to_groove, groove_to_peptide}, ...}
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            peptide_tokens, mhc_tokens,
            peptide_mask=peptide_mask,
            mhc_mask=mhc_mask,
            return_attention=True
        )
    return outputs['attention_info']


def visualize_region_attention(
    attention_info: Dict,
    sample_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize region identification attention (Level 1 of groove extraction).
    
    Shows which MHC regions are identified by each region query.
    """
    region_attn = attention_info['groove_extraction']['region_attention']
    # Shape: [batch, num_heads, num_regions, mhc_len]
    
    attn = region_attn[sample_idx].mean(dim=0).cpu().numpy()  # Average over heads
    # Shape: [num_regions, mhc_len]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    sns.heatmap(
        attn,
        cmap='Blues',
        ax=ax,
        xticklabels=50,
        yticklabels=[f'Region {i+1}' for i in range(attn.shape[0])]
    )
    ax.set_xlabel('MHC Position')
    ax.set_ylabel('Region Query')
    ax.set_title('Region Identification Attention\n(Which MHC regions are identified)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}" if save_path else "  (not saved)")
    

def visualize_groove_attention(
    attention_info: Dict,
    sample_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize groove extraction attention (Level 2 of groove extraction).
    
    Shows which MHC positions are extracted into each groove token.
    """
    groove_attn = attention_info['groove_extraction']['groove_attention']
    # Shape: [batch, num_heads, num_groove_tokens, mhc_len]
    
    attn = groove_attn[sample_idx].mean(dim=0).cpu().numpy()  # Average over heads
    # Shape: [num_groove_tokens, mhc_len]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(
        attn,
        cmap='Greens',
        ax=ax,
        xticklabels=50,
        yticklabels=10
    )
    ax.set_xlabel('MHC Position')
    ax.set_ylabel('Groove Token')
    ax.set_title('Groove Extraction Attention\n(Which MHC positions contribute to each groove token)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}" if save_path else "  (not saved)")


def visualize_position_importance(
    attention_info: Dict,
    peptide_sequence: Optional[str] = None,
    sample_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize peptide position importance scores.
    
    Shows which positions (anchor positions) are considered most important.
    
    Note: ESM encoder now handles BOS/EOS stripping internally.
    """
    importance = attention_info['position_importance']
    # Shape: [batch, pep_len]
    
    imp = importance[sample_idx].cpu().numpy()
    
    # Note: ESM encoder now strips BOS and zeros EOS internally,
    # so the importance scores match the actual peptide length.
    # However, zeroed EOS position may still be present, we handle that below.
    
    pep_len = len(imp)
    
    # If peptide_sequence is provided and shorter than imp (ESM has EOS zeroed at end),
    # we truncate imp to match the actual sequence
    if peptide_sequence and len(peptide_sequence) < pep_len:
        imp = imp[:len(peptide_sequence)]
        pep_len = len(imp)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if peptide_sequence and len(peptide_sequence) == pep_len:
        labels = [f'{aa}\n(P{i+1})' for i, aa in enumerate(peptide_sequence)]
    else:
        labels = [f'P{i+1}' for i in range(pep_len)]
    
    colors = plt.cm.Reds(imp / (imp.max() + 1e-8))
    bars = ax.bar(range(pep_len), imp, color=colors)
    
    ax.set_xticks(range(pep_len))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Position Importance')
    ax.set_title('Peptide Position Importance\n(Anchor positions like P2, P9 typically have high importance)')
    ax.set_ylim(0, max(1.1, imp.max() * 1.1))
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, imp)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}" if save_path else "  (not saved)")


def visualize_fusion_attention(
    attention_info: Dict,
    layer_idx: int = -1,
    sample_idx: int = 0,
    peptide_sequence: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize peptide-groove fusion cross-attention.
    
    Shows bidirectional attention:
    - Peptide → Groove: What groove regions does each peptide position attend to?
    - Groove → Peptide: What peptide positions does each groove region attend to?
    
    Note: ESM encoder handles BOS/EOS stripping internally.
    """
    fusion_info = attention_info['fusion']
    layer_keys = sorted(fusion_info.keys())
    
    if layer_idx == -1:
        layer_key = layer_keys[-1]  # Last layer
    else:
        layer_key = layer_keys[layer_idx]
    
    layer_attn = fusion_info[layer_key]
    
    pep_to_groove = layer_attn['peptide_to_groove'][sample_idx].mean(dim=0).cpu().numpy()
    groove_to_pep = layer_attn['groove_to_peptide'][sample_idx].mean(dim=0).cpu().numpy()
    
    # Truncate to peptide_sequence length if provided (to handle zeroed EOS position)
    if peptide_sequence:
        actual_len = len(peptide_sequence)
        if pep_to_groove.shape[0] > actual_len:
            pep_to_groove = pep_to_groove[:actual_len, :]
        if groove_to_pep.shape[1] > actual_len:
            groove_to_pep = groove_to_pep[:, :actual_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Peptide → Groove
    ax1 = axes[0]
    sns.heatmap(
        pep_to_groove,
        cmap='Purples',
        ax=ax1,
        xticklabels=10,
        yticklabels=True
    )
    if peptide_sequence:
        pep_labels = [f'{aa}(P{i+1})' for i, aa in enumerate(peptide_sequence[:pep_to_groove.shape[0]])]
        ax1.set_yticklabels(pep_labels)
    ax1.set_xlabel('Groove Position')
    ax1.set_ylabel('Peptide Position')
    ax1.set_title(f'Peptide → Groove Attention ({layer_key})\n(What groove regions each peptide position attends to)')
    
    # Groove → Peptide
    ax2 = axes[1]
    sns.heatmap(
        groove_to_pep,
        cmap='Oranges',
        ax=ax2,
        xticklabels=True,
        yticklabels=10
    )
    if peptide_sequence:
        ax2.set_xticklabels([f'{aa}' for aa in peptide_sequence[:groove_to_pep.shape[1]]])
    ax2.set_xlabel('Peptide Position')
    ax2.set_ylabel('Groove Position')
    ax2.set_title(f'Groove → Peptide Attention ({layer_key})\n(What peptide positions each groove region attends to)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}" if save_path else "  (not saved)")


def visualize_all_fusion_layers(
    attention_info: Dict,
    sample_idx: int = 0,
    peptide_sequence: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize attention evolution across all fusion layers.
    
    Note: ESM encoder handles BOS/EOS stripping internally.
    """
    fusion_info = attention_info['fusion']
    layer_keys = sorted(fusion_info.keys())
    num_layers = len(layer_keys)
    
    fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 10))
    
    for i, layer_key in enumerate(layer_keys):
        layer_attn = fusion_info[layer_key]
        
        pep_to_groove = layer_attn['peptide_to_groove'][sample_idx].mean(dim=0).cpu().numpy()
        groove_to_pep = layer_attn['groove_to_peptide'][sample_idx].mean(dim=0).cpu().numpy()
        
        # Truncate to peptide_sequence length if provided
        if peptide_sequence:
            actual_len = len(peptide_sequence)
            if pep_to_groove.shape[0] > actual_len:
                pep_to_groove = pep_to_groove[:actual_len, :]
            if groove_to_pep.shape[1] > actual_len:
                groove_to_pep = groove_to_pep[:, :actual_len]
        
        # Peptide → Groove
        ax1 = axes[0, i]
        sns.heatmap(pep_to_groove, cmap='Purples', ax=ax1, cbar=False)
        ax1.set_title(f'Layer {i}: P→G')
        ax1.set_xlabel('Groove')
        ax1.set_ylabel('Peptide' if i == 0 else '')
        
        # Groove → Peptide
        ax2 = axes[1, i]
        sns.heatmap(groove_to_pep, cmap='Oranges', ax=ax2, cbar=False)
        ax2.set_title(f'Layer {i}: G→P')
        ax2.set_xlabel('Peptide')
        ax2.set_ylabel('Groove' if i == 0 else '')
    
    fig.suptitle('Attention Evolution Across Fusion Layers', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}" if save_path else "  (not saved)")


def comprehensive_attention_visualization(
    model: DinoMHC,
    peptide_tokens: torch.Tensor,
    mhc_tokens: torch.Tensor,
    peptide_sequence: Optional[str] = None,
    peptide_mask: Optional[torch.Tensor] = None,
    mhc_mask: Optional[torch.Tensor] = None,
    save_prefix: Optional[str] = None
):
    """
    Generate comprehensive visualization of all attention mechanisms.
    
    Note: ESM encoder handles BOS/EOS stripping internally.
    """
    print("=" * 70)
    print("COMPREHENSIVE ATTENTION VISUALIZATION")
    print("=" * 70)
    
    # Get attention maps
    attention_info = get_attention_maps(
        model, peptide_tokens, mhc_tokens,
        peptide_mask=peptide_mask,
        mhc_mask=mhc_mask
    )
    
    print("\n1. Region Identification Attention")
    print("-" * 40)
    visualize_region_attention(
        attention_info, 
        save_path=f"{save_prefix}_region_attention.png" if save_prefix else None
    )
    
    print("\n2. Groove Extraction Attention")
    print("-" * 40)
    visualize_groove_attention(
        attention_info,
        save_path=f"{save_prefix}_groove_attention.png" if save_prefix else None
    )
    
    print("\n3. Peptide Position Importance")
    print("-" * 40)
    visualize_position_importance(
        attention_info,
        peptide_sequence=peptide_sequence,
        save_path=f"{save_prefix}_position_importance.png" if save_prefix else None
    )
    
    print("\n4. Peptide-Groove Fusion Attention (Final Layer)")
    print("-" * 40)
    visualize_fusion_attention(
        attention_info,
        layer_idx=-1,
        peptide_sequence=peptide_sequence,
        save_path=f"{save_prefix}_fusion_attention.png" if save_prefix else None
    )
    
    print("\n5. Attention Evolution Across Fusion Layers")
    print("-" * 40)
    visualize_all_fusion_layers(
        attention_info,
        peptide_sequence=peptide_sequence,
        save_path=f"{save_prefix}_fusion_evolution.png" if save_prefix else None
    )
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


def create_attention_summary_figure(
    attention_info: Dict,
    peptide_sequence: Optional[str] = None,
    sample_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Create a single summary figure with all attention visualizations.
    
    Note: ESM encoder handles BOS/EOS stripping internally.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Region Attention (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    region_attn = attention_info['groove_extraction']['region_attention']
    attn = region_attn[sample_idx].mean(dim=0).cpu().numpy()
    sns.heatmap(attn, cmap='Blues', ax=ax1, xticklabels=50, 
                yticklabels=[f'R{i+1}' for i in range(attn.shape[0])])
    ax1.set_xlabel('MHC Position')
    ax1.set_ylabel('Region')
    ax1.set_title('1. Region Identification')
    
    # 2. Groove Extraction (top-center & top-right)
    ax2 = fig.add_subplot(gs[0, 1:])
    groove_attn = attention_info['groove_extraction']['groove_attention']
    attn = groove_attn[sample_idx].mean(dim=0).cpu().numpy()
    sns.heatmap(attn, cmap='Greens', ax=ax2, xticklabels=50, yticklabels=10)
    ax2.set_xlabel('MHC Position')
    ax2.set_ylabel('Groove Token')
    ax2.set_title('2. Groove Extraction')
    
    # 3. Position Importance (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    importance = attention_info['position_importance']
    imp = importance[sample_idx].cpu().numpy()
    
    # Truncate to peptide_sequence length if provided
    if peptide_sequence and len(imp) > len(peptide_sequence):
        imp = imp[:len(peptide_sequence)]
    
    pep_len = len(imp)
    colors = plt.cm.Reds(imp / (imp.max() + 1e-8))
    ax3.bar(range(pep_len), imp, color=colors)
    if peptide_sequence:
        ax3.set_xticks(range(pep_len))
        ax3.set_xticklabels([f'{aa}' for aa in peptide_sequence[:pep_len]])
    ax3.set_ylabel('Importance')
    ax3.set_title('3. Position Importance')
    
    # 4 & 5. Fusion Attention (middle-center & middle-right)
    fusion_info = attention_info['fusion']
    layer_keys = sorted(fusion_info.keys())
    last_layer = fusion_info[layer_keys[-1]]
    
    ax4 = fig.add_subplot(gs[1, 1])
    pep_to_groove = last_layer['peptide_to_groove'][sample_idx].mean(dim=0).cpu().numpy()
    if peptide_sequence and pep_to_groove.shape[0] > len(peptide_sequence):
        pep_to_groove = pep_to_groove[:len(peptide_sequence), :]
    sns.heatmap(pep_to_groove, cmap='Purples', ax=ax4, xticklabels=10)
    ax4.set_xlabel('Groove')
    ax4.set_ylabel('Peptide')
    ax4.set_title('4. Peptide→Groove (final)')
    
    ax5 = fig.add_subplot(gs[1, 2])
    groove_to_pep = last_layer['groove_to_peptide'][sample_idx].mean(dim=0).cpu().numpy()
    if peptide_sequence and groove_to_pep.shape[1] > len(peptide_sequence):
        groove_to_pep = groove_to_pep[:, :len(peptide_sequence)]
    sns.heatmap(groove_to_pep, cmap='Oranges', ax=ax5, yticklabels=10)
    ax5.set_xlabel('Peptide')
    ax5.set_ylabel('Groove')
    ax5.set_title('5. Groove→Peptide (final)')
    
    # 6. Attention Evolution (bottom row)
    num_layers = len(layer_keys)
    for i, layer_key in enumerate(layer_keys):
        if i >= 3:  # Only plot first 3 layers in this row
            break
        ax = fig.add_subplot(gs[2, i])
        layer_attn = fusion_info[layer_key]
        
        # P→G: For each peptide position, show max attention it gives to any groove position
        # This shows which peptide positions are most "focused" on specific groove regions
        pep_to_groove = layer_attn['peptide_to_groove'][sample_idx].mean(dim=0)  # [pep_len, groove_len]
        
        # G→P: For each peptide position, show max attention it receives from any groove position  
        # This shows which peptide positions are most "attended to" by groove
        groove_to_pep = layer_attn['groove_to_peptide'][sample_idx].mean(dim=0)  # [groove_len, pep_len]
        
        # Truncate to peptide_sequence length if provided
        if peptide_sequence:
            actual_len = len(peptide_sequence)
            if pep_to_groove.shape[0] > actual_len:
                pep_to_groove = pep_to_groove[:actual_len, :]
            if groove_to_pep.shape[1] > actual_len:
                groove_to_pep = groove_to_pep[:, :actual_len]
        
        pep_max_attn = pep_to_groove.max(dim=1).values.cpu().numpy()  # [pep_len]
        pep_max_received = groove_to_pep.max(dim=0).values.cpu().numpy()  # [pep_len]
        
        x = np.arange(len(pep_max_attn))
        width = 0.35
        ax.bar(x - width/2, pep_max_attn, width, alpha=0.8, label='P→G (max)', color='purple')
        ax.bar(x + width/2, pep_max_received, width, alpha=0.8, label='G→P (max)', color='orange')
        ax.set_title(f'Layer {i}')
        ax.set_xlabel('Peptide Pos')
        ax.set_ylabel('Max Attn')
        ax.set_xticks(x)
        if i == 0:
            ax.legend(fontsize=7)
    
    fig.suptitle('DinoMHC Attention Summary', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}" if save_path else "  (not saved)")


# =============================================================================
# DEMO
# =============================================================================

def run_visualization(encoder_type: str = 'embedding', save_prefix: str = 'attention'):
    """
    Run attention visualization with specified encoder type.
    
    Args:
        encoder_type: 'embedding', 'esm2', or 'esm2_shared'
        save_prefix: Prefix for saved figure files
    """
    print("=" * 70)
    print(f"DINOMHC ATTENTION VISUALIZATION")
    print(f"Encoder: {encoder_type}")
    print("=" * 70)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Example sequences
    peptide_sequence = "YLQPRTFLL"  # Example HLA-A*02:01 binder (9-mer)
    mhc_sequence = "MAVMAPRTLLLLLSGALALTQTWAG" + "A" * 75  # 100 AA for demo
    
    print(f"\nPeptide: {peptide_sequence} (len={len(peptide_sequence)})")
    print(f"MHC length: {len(mhc_sequence)}")
    
    # Create model based on encoder type
    if encoder_type in ['esm2', 'esm2_shared']:
        if not HAS_ESM:
            raise ImportError("ESM package required. Install with: pip install fair-esm")
        
        config = {
            'dim': 320,
            'encoder_type': encoder_type,
            'esm_model_name': 'esm2_t6_8M_UR50D',
            'esm_unfreeze_layers': 0,  # Frozen for inference
            'num_groove_tokens': 50,
            'num_fusion_layers': 4,
            'num_geometry_blocks': 2,
            'num_heads': 4,
            'dropout': 0.0,
            'max_peptide_length': 15,
            'task_head': 'presentation'
        }
        
        # ESM tokenization
        _, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')
        batch_converter = alphabet.get_batch_converter()
        padding_idx = alphabet.padding_idx
        
        # Tokenize
        pep_data = [("peptide", peptide_sequence)]
        mhc_data = [("mhc", mhc_sequence)]
        
        _, _, peptide_tokens = batch_converter(pep_data)
        _, _, mhc_tokens = batch_converter(mhc_data)
        
        # Create masks
        peptide_mask = peptide_tokens != padding_idx
        mhc_mask = mhc_tokens != padding_idx
        
    else:
        config = {
            'dim': 128,
            'encoder_type': 'embedding',
            'num_groove_tokens': 50,
            'num_fusion_layers': 4,
            'num_geometry_blocks': 2,
            'num_heads': 4,
            'dropout': 0.0,
            'max_peptide_length': 15,
            'task_head': 'presentation'
        }
        
        # Simple tokenization
        aa_to_idx = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        peptide_tokens = torch.tensor([[aa_to_idx.get(aa, 0) for aa in peptide_sequence]])
        mhc_tokens = torch.tensor([[aa_to_idx.get(aa, 0) for aa in mhc_sequence]])
        
        peptide_mask = torch.ones_like(peptide_tokens, dtype=torch.bool)
        mhc_mask = torch.ones_like(mhc_tokens, dtype=torch.bool)
    
    # Create and load model
    print(f"\nCreating model with config:")
    print(f"  dim: {config['dim']}")
    print(f"  encoder_type: {config['encoder_type']}")
    print(f"  fusion_layers: {config['num_fusion_layers']}")
    
    model = DinoMHC(config)
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Move tensors to device
    peptide_tokens = peptide_tokens.to(device)
    mhc_tokens = mhc_tokens.to(device)
    peptide_mask = peptide_mask.to(device)
    mhc_mask = mhc_mask.to(device)
    
    print(f"\nInput shapes:")
    print(f"  Peptide tokens: {peptide_tokens.shape}")
    print(f"  MHC tokens: {mhc_tokens.shape}")
    
    # Run visualization
    full_save_prefix = f"{save_prefix}_{encoder_type}"
    
    comprehensive_attention_visualization(
        model,
        peptide_tokens,
        mhc_tokens,
        peptide_sequence=peptide_sequence,
        peptide_mask=peptide_mask,
        mhc_mask=mhc_mask,
        save_prefix=full_save_prefix
    )
    
    # Create summary figure
    print("\nCreating summary figure...")
    attention_info = get_attention_maps(
        model, peptide_tokens, mhc_tokens,
        peptide_mask=peptide_mask, mhc_mask=mhc_mask
    )
    create_attention_summary_figure(
        attention_info,
        peptide_sequence=peptide_sequence,
        save_path=f'{full_save_prefix}_summary.png'
    )
    
    print(f"\nSaved figures with prefix: {full_save_prefix}_*.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DinoMHC Attention Visualization')
    parser.add_argument('--encoder', type=str, default='embedding',
                        choices=['embedding', 'esm2', 'esm2_shared'],
                        help='Encoder type to use')
    parser.add_argument('--save-prefix', type=str, default='attention_viz',
                        help='Prefix for saved figure files')
    args = parser.parse_args()
    
    run_visualization(encoder_type=args.encoder, save_prefix=args.save_prefix)
