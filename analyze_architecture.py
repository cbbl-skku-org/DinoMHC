#!/usr/bin/env python
"""
DinoMHC Architecture Analysis

Provides detailed analysis of:
1. Model size (parameters)
2. Memory footprint
3. Time complexity analysis
4. FLOPs estimation
5. Layer-by-layer breakdown
"""

import torch
import torch.nn as nn
import time
import sys
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict

sys.path.insert(0, '/home/duongtt/Workdir/MHC/digermhc/src')
from model import DinoMHC


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count total parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    """Format parameter count with appropriate suffix."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def get_module_params(module: nn.Module) -> Dict[str, int]:
    """Get parameter counts for immediate children of a module."""
    params = {}
    for name, child in module.named_children():
        params[name] = count_parameters(child)
    return params


def analyze_model_architecture(model: DinoMHC) -> Dict:
    """
    Comprehensive architecture analysis.
    """
    config = model.config
    dim = config['dim']
    num_heads = config['num_heads']
    num_groove_tokens = config['num_groove_tokens']
    num_fusion_layers = config['num_fusion_layers']
    num_geometry_blocks = config['num_geometry_blocks']
    max_pep_len = config['max_peptide_length']
    
    analysis = {
        'config': config,
        'total_params': count_parameters(model),
        'trainable_params': count_parameters(model, trainable_only=True),
        'modules': {}
    }
    
    # Analyze each major module
    modules_info = {
        'peptide_encoder': model.peptide_encoder,
        'mhc_encoder': model.mhc_encoder,
        'groove_extractor': model.groove_extractor,
        'position_encoder': model.position_encoder,
        'groove_peptide_fusion': model.groove_peptide_fusion,
        'interface_geometry': model.interface_geometry,
        'task_head': model.task_head,
    }
    
    for name, module in modules_info.items():
        params = count_parameters(module)
        analysis['modules'][name] = {
            'params': params,
            'params_formatted': format_params(params),
            'percentage': 100 * params / analysis['total_params']
        }
    
    return analysis


def estimate_flops(
    config: Dict,
    pep_len: int = 9,
    mhc_len: int = 180
) -> Dict:
    """
    Estimate FLOPs (Floating Point Operations) for forward pass.
    
    This is a rough estimation based on major operations.
    """
    dim = config['dim']
    num_heads = config['num_heads']
    head_dim = dim // num_heads
    num_groove_tokens = config['num_groove_tokens']
    num_fusion_layers = config['num_fusion_layers']
    num_geometry_blocks = config['num_geometry_blocks']
    ff_dim = dim * 4  # Typical feed-forward expansion
    
    flops = {}
    
    # 1. Sequence Encoding (Embedding + Linear)
    flops['encoding'] = {
        'peptide': pep_len * dim + pep_len * dim * dim,  # Embedding lookup + Linear
        'mhc': mhc_len * dim + mhc_len * dim * dim
    }
    
    # 2. Hierarchical Groove Extractor
    num_regions = 3
    # Region attention: Q*K^T + softmax*V
    region_attn_flops = 2 * num_regions * mhc_len * dim  # QKV projections
    region_attn_flops += num_heads * num_regions * mhc_len  # QK^T
    region_attn_flops += num_heads * num_regions * mhc_len  # softmax*V
    
    # Groove attention (similar)
    groove_attn_flops = 2 * num_groove_tokens * mhc_len * dim
    groove_attn_flops += num_heads * num_groove_tokens * mhc_len * 2
    
    flops['groove_extractor'] = region_attn_flops + groove_attn_flops
    
    # 3. Position-Aware Peptide Encoder
    # Self-attention + importance prediction
    pos_encoder_flops = 3 * pep_len * dim * dim  # QKV
    pos_encoder_flops += num_heads * pep_len * pep_len * 2  # Attention
    pos_encoder_flops += pep_len * dim  # Importance predictor
    flops['position_encoder'] = pos_encoder_flops
    
    # 4. Groove-Peptide Fusion (per layer)
    fusion_layer_flops = 0
    
    # Peptide self-attention
    fusion_layer_flops += 3 * pep_len * dim * dim  # QKV
    fusion_layer_flops += num_heads * pep_len * pep_len * 2
    
    # Groove self-attention
    fusion_layer_flops += 3 * num_groove_tokens * dim * dim
    fusion_layer_flops += num_heads * num_groove_tokens * num_groove_tokens * 2
    
    # Peptide→Groove cross-attention
    fusion_layer_flops += 3 * pep_len * dim * dim + 3 * num_groove_tokens * dim * dim
    fusion_layer_flops += num_heads * pep_len * num_groove_tokens * 2
    # + Geometric bias
    fusion_layer_flops += pep_len * num_groove_tokens * dim * 2  # Pair encoding
    
    # Groove→Peptide cross-attention (similar)
    fusion_layer_flops += 3 * num_groove_tokens * dim * dim + 3 * pep_len * dim * dim
    fusion_layer_flops += num_heads * num_groove_tokens * pep_len * 2
    fusion_layer_flops += num_groove_tokens * pep_len * dim * 2
    
    # FFNs
    fusion_layer_flops += 2 * pep_len * dim * ff_dim  # Peptide FFN
    fusion_layer_flops += 2 * num_groove_tokens * dim * ff_dim  # Groove FFN
    
    flops['groove_peptide_fusion'] = fusion_layer_flops * num_fusion_layers
    
    # 5. Interface Geometry Module
    max_len = max(pep_len, num_groove_tokens)
    pair_size = max_len * max_len
    
    # Pair representation creation
    geom_flops = pep_len * num_groove_tokens * dim * 2 * dim  # Concat + project
    
    # Per geometry block
    geom_block_flops = 0
    
    # Triangle attention (starting + ending)
    geom_block_flops += 2 * (3 * pair_size * dim * dim)  # QKV for both modes
    geom_block_flops += 2 * (num_heads // 2 * max_len * max_len * max_len)  # Attention
    
    # Triangle multiplication (outgoing + incoming)
    geom_block_flops += 2 * (2 * pair_size * dim * dim)  # Left/right proj
    geom_block_flops += 2 * (max_len * max_len * max_len * dim)  # Einsum
    
    # Pair transition (FFN)
    geom_block_flops += 2 * pair_size * dim * ff_dim
    
    geom_flops += geom_block_flops * num_geometry_blocks
    flops['interface_geometry'] = geom_flops
    
    # 6. Task Head
    flops['task_head'] = dim * dim // 2 + dim // 2 * dim // 4 + dim // 4
    
    # Total
    total = sum(
        v if isinstance(v, (int, float)) else sum(v.values()) 
        for v in flops.values()
    )
    flops['total'] = total
    
    return flops


def format_flops(flops: int) -> str:
    """Format FLOPs with appropriate suffix."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def measure_inference_time(
    model: DinoMHC,
    pep_len: int = 9,
    mhc_len: int = 180,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_runs: int = 5,
    device: str = 'cpu'
) -> Dict:
    """
    Measure actual inference time.
    """
    model = model.to(device)
    model.eval()
    
    peptide_tokens = torch.randint(1, 20, (batch_size, pep_len)).to(device)
    mhc_tokens = torch.randint(1, 20, (batch_size, mhc_len)).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(peptide_tokens, mhc_tokens)
    
    # Synchronize if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(peptide_tokens, mhc_tokens)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    return {
        'mean_ms': 1000 * sum(times) / len(times),
        'std_ms': 1000 * (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'min_ms': 1000 * min(times),
        'max_ms': 1000 * max(times),
        'throughput': batch_size / (sum(times) / len(times))  # samples/sec
    }


def analyze_time_complexity(config: Dict) -> Dict:
    """
    Theoretical time complexity analysis.
    
    Let:
    - P = peptide length
    - M = MHC length  
    - G = num_groove_tokens
    - D = embedding dimension
    - L = num_fusion_layers
    - B = num_geometry_blocks
    - H = num_heads
    """
    complexity = {}
    
    complexity['encoding'] = {
        'description': 'Embedding + Linear projection',
        'complexity': 'O(P*D + M*D)',
        'dominant': False
    }
    
    complexity['groove_extractor'] = {
        'description': 'Hierarchical attention over MHC',
        'complexity': 'O(G*M*D + M*D)',
        'dominant': 'For long MHC sequences'
    }
    
    complexity['position_encoder'] = {
        'description': 'Self-attention on peptide',
        'complexity': 'O(P²*D)',
        'dominant': False
    }
    
    complexity['groove_peptide_fusion'] = {
        'description': 'Bidirectional cross-attention (L layers)',
        'complexity': 'O(L * (P²*D + G²*D + P*G*D))',
        'dominant': 'Scales with L and P*G'
    }
    
    complexity['interface_geometry'] = {
        'description': 'Triangle attention/multiplication (B blocks)',
        'complexity': 'O(B * max(P,G)³ * D)',
        'dominant': '⚠️ CUBIC in sequence length!'
    }
    
    complexity['overall'] = {
        'description': 'Total model complexity',
        'complexity': 'O(B * max(P,G)³ * D + L * P*G*D)',
        'bottleneck': 'Interface Geometry (triangle ops)'
    }
    
    return complexity


def analyze_memory_footprint(
    model: DinoMHC,
    pep_len: int = 9,
    mhc_len: int = 180,
    batch_size: int = 1
) -> Dict:
    """
    Estimate memory footprint.
    """
    config = model.config
    dim = config['dim']
    num_groove_tokens = config['num_groove_tokens']
    num_fusion_layers = config['num_fusion_layers']
    num_geometry_blocks = config['num_geometry_blocks']
    
    # Model parameters (float32 = 4 bytes)
    param_memory = count_parameters(model) * 4
    
    # Activations during forward pass (rough estimates)
    activations = {}
    
    # Embeddings
    activations['peptide_emb'] = batch_size * pep_len * dim * 4
    activations['mhc_emb'] = batch_size * mhc_len * dim * 4
    
    # Groove extraction
    activations['groove_emb'] = batch_size * num_groove_tokens * dim * 4
    activations['region_attn'] = batch_size * 8 * 3 * mhc_len * 4  # heads * regions * mhc_len
    activations['groove_attn'] = batch_size * 8 * num_groove_tokens * mhc_len * 4
    
    # Fusion (per layer, largest intermediate)
    fusion_attn = batch_size * 8 * max(pep_len, num_groove_tokens) ** 2 * 4
    activations['fusion_layers'] = fusion_attn * num_fusion_layers
    
    # Geometric bias computation
    activations['geometric_bias'] = batch_size * pep_len * num_groove_tokens * dim * 2 * 4
    
    # Interface geometry (largest!)
    max_len = max(pep_len, num_groove_tokens)
    pair_rep_size = batch_size * max_len * max_len * dim * 4
    triangle_attn_size = batch_size * (8 // 2) * max_len * max_len * 4
    activations['pair_representation'] = pair_rep_size
    activations['triangle_attention'] = triangle_attn_size * 2 * num_geometry_blocks
    
    total_activations = sum(activations.values())
    
    return {
        'parameters_mb': param_memory / (1024 ** 2),
        'activations_mb': total_activations / (1024 ** 2),
        'total_mb': (param_memory + total_activations) / (1024 ** 2),
        'breakdown': {k: v / (1024 ** 2) for k, v in activations.items()}
    }


def print_architecture_report(model: DinoMHC, pep_len: int = 9, mhc_len: int = 180):
    """
    Print comprehensive architecture report.
    """
    config = model.config
    
    print("=" * 80)
    print("DINOMHC ARCHITECTURE ANALYSIS")
    print("=" * 80)
    
    # 1. Configuration
    print("\n" + "─" * 80)
    print("1. MODEL CONFIGURATION")
    print("─" * 80)
    for key, value in config.items():
        print(f"   {key:25s}: {value}")
    
    # 2. Parameter Counts
    print("\n" + "─" * 80)
    print("2. PARAMETER COUNTS")
    print("─" * 80)
    
    analysis = analyze_model_architecture(model)
    print(f"\n   Total Parameters:     {format_params(analysis['total_params']):>12s} ({analysis['total_params']:,})")
    print(f"   Trainable Parameters: {format_params(analysis['trainable_params']):>12s}")
    
    print("\n   Module Breakdown:")
    print("   " + "-" * 60)
    for name, info in analysis['modules'].items():
        print(f"   {name:30s} {info['params_formatted']:>10s} ({info['percentage']:5.1f}%)")
    
    # 3. FLOPs Estimation
    print("\n" + "─" * 80)
    print(f"3. FLOPs ESTIMATION (pep_len={pep_len}, mhc_len={mhc_len})")
    print("─" * 80)
    
    flops = estimate_flops(config, pep_len, mhc_len)
    print(f"\n   Total FLOPs: {format_flops(flops['total'])}")
    print("\n   Breakdown:")
    print("   " + "-" * 60)
    for name, value in flops.items():
        if name != 'total':
            if isinstance(value, dict):
                total = sum(value.values())
                print(f"   {name:30s} {format_flops(total):>15s}")
            else:
                print(f"   {name:30s} {format_flops(value):>15s}")
    
    # 4. Time Complexity
    print("\n" + "─" * 80)
    print("4. TIME COMPLEXITY ANALYSIS")
    print("─" * 80)
    
    complexity = analyze_time_complexity(config)
    for name, info in complexity.items():
        print(f"\n   {name}:")
        print(f"      Description: {info['description']}")
        print(f"      Complexity:  {info['complexity']}")
        if 'dominant' in info:
            print(f"      Note:        {info['dominant']}")
        if 'bottleneck' in info:
            print(f"      Bottleneck:  {info['bottleneck']}")
    
    # 5. Memory Footprint
    print("\n" + "─" * 80)
    print(f"5. MEMORY FOOTPRINT (batch_size=1, pep_len={pep_len}, mhc_len={mhc_len})")
    print("─" * 80)
    
    memory = analyze_memory_footprint(model, pep_len, mhc_len, batch_size=1)
    print(f"\n   Model Parameters:  {memory['parameters_mb']:8.2f} MB")
    print(f"   Activations:       {memory['activations_mb']:8.2f} MB")
    print(f"   Total (estimate):  {memory['total_mb']:8.2f} MB")
    
    print("\n   Activation Breakdown:")
    print("   " + "-" * 50)
    for name, mb in sorted(memory['breakdown'].items(), key=lambda x: -x[1]):
        print(f"   {name:30s} {mb:8.2f} MB")
    
    # 6. Inference Time
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "─" * 80)
    print(f"6. INFERENCE TIME ({device.upper()})")
    print("─" * 80)
    
    print(f"\n   Measuring inference time (5 runs on {device})...")
    timing = measure_inference_time(model, pep_len, mhc_len, batch_size=1, device=device)
    print(f"   Mean:       {timing['mean_ms']:8.2f} ms")
    print(f"   Std:        {timing['std_ms']:8.2f} ms")
    print(f"   Min:        {timing['min_ms']:8.2f} ms")
    print(f"   Max:        {timing['max_ms']:8.2f} ms")
    print(f"   Throughput: {timing['throughput']:8.1f} samples/sec")
    
    # Different batch sizes
    print("\n   Batch size scaling:")
    for bs in [1, 8, 32]:
        timing = measure_inference_time(model, pep_len, mhc_len, batch_size=bs, device=device, num_runs=5)
        print(f"   Batch={bs:3d}: {timing['mean_ms']:8.2f} ms ({timing['throughput']:6.1f} samples/sec)")
    
    # 7. Architecture Diagram
    print("\n" + "─" * 80)
    print("7. ARCHITECTURE FLOW")
    print("─" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                        INPUT                                     │
    │  Peptide Tokens [B, P]              MHC Tokens [B, M]           │
    └─────────────────┬─────────────────────────────┬─────────────────┘
                      │                             │
                      ▼                             ▼
    ┌─────────────────────────┐       ┌─────────────────────────────┐
    │   Peptide Encoder       │       │      MHC Encoder            │
    │   [B, P, D]             │       │      [B, M, D]              │
    └───────────┬─────────────┘       └─────────────┬───────────────┘
                │                                   │
                │         ┌─────────────────────────┘
                │         │
                │         ▼
                │    ┌────────────────────────────────────────────┐
                │    │     Hierarchical Groove Extractor          │
                │    │  ┌─────────────────────────────────────┐  │
                │    │  │  Level 1: Region Identification     │  │
                │    │  │  [B, 3, M] region attention         │  │
                │    │  └──────────────┬──────────────────────┘  │
                │    │                 │                          │
                │    │  ┌──────────────▼──────────────────────┐  │
                │    │  │  Level 2: Groove Extraction         │  │
                │    │  │  [B, G, D] groove features          │  │
                │    │  └─────────────────────────────────────┘  │
                │    └─────────────────┬──────────────────────────┘
                │                      │
                ▼                      ▼
    ┌───────────────────────┐    ┌───────────────────────────────┐
    │  Position-Aware       │    │   Groove Features             │
    │  Peptide Encoder      │    │   [B, G, D]                   │
    │  [B, P, D]            │    │                               │
    └───────────┬───────────┘    └─────────────┬─────────────────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                  Groove-Peptide Fusion                        │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │  Layer 1..L: Bidirectional Cross-Attention             │  │
    │  │  • Peptide Self-Attention                              │  │
    │  │  • Groove Self-Attention                               │  │
    │  │  • Peptide → Groove (with geometric bias)              │  │
    │  │  • Groove → Peptide (with geometric bias)              │  │
    │  │  • Feed-Forward Networks                               │  │
    │  └────────────────────────────────────────────────────────┘  │
    │  Output: Peptide [B, P, D], Groove [B, G, D]                 │
    └──────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │               Interface Geometry Module                       │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │  Pair Representation: [B, max(P,G), max(P,G), D]       │  │
    │  │  (padded to square with masking)                       │  │
    │  └───────────────────────┬────────────────────────────────┘  │
    │                          │                                    │
    │  ┌───────────────────────▼────────────────────────────────┐  │
    │  │  Geometry Blocks (1..B):                               │  │
    │  │  • Triangle Attention (starting)                       │  │
    │  │  • Triangle Attention (ending)                         │  │
    │  │  • Triangle Multiplication (outgoing)                  │  │
    │  │  • Triangle Multiplication (incoming)                  │  │
    │  │  • Pair Transition (FFN)                               │  │
    │  └───────────────────────┬────────────────────────────────┘  │
    │                          │                                    │
    │  Output: Interface Rep [B, D], Pair Rep [B, P, G, D]         │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                      Task Head                                │
    │  • Presentation: Sigmoid → [B, 1] (binding probability)      │
    │  • Affinity: Linear → [B, 1] (IC50/KD value)                 │
    │  • Contact: Pair → [B, P, G] (contact map)                   │
    └──────────────────────────────────────────────────────────────┘
    """)
    
    print("=" * 80)
    print("END OF ARCHITECTURE REPORT")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Create models with different configurations
    
    # Small model (for quick experiments)
    small_config = {
        'dim': 128,
        'num_groove_tokens': 50,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    # Medium model (balanced)
    medium_config = {
        'dim': 256,
        'num_groove_tokens': 60,
        'num_fusion_layers': 4,
        'num_geometry_blocks': 3,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    # Large model (high capacity)
    large_config = {
        'dim': 320,
        'num_groove_tokens': 60,
        'num_fusion_layers': 6,
        'num_geometry_blocks': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    print("\n" + "=" * 80)
    print("COMPARING MODEL SIZES")
    print("=" * 80)
    
    configs = [
        ('Small', small_config),
        ('Medium', medium_config),
        ('Large', large_config)
    ]
    
    print(f"\n{'Model':<10} {'Params':>12} {'FLOPs':>15} {'Memory (MB)':>12}")
    print("-" * 55)
    
    for name, cfg in configs:
        model = DinoMHC(cfg)
        params = count_parameters(model)
        flops = estimate_flops(cfg, pep_len=9, mhc_len=180)
        memory = analyze_memory_footprint(model, pep_len=9, mhc_len=180)
        print(f"{name:<10} {format_params(params):>12} {format_flops(flops['total']):>15} {memory['total_mb']:>12.1f}")
    
    # Detailed report for medium model
    print("\n")
    model = DinoMHC(medium_config)
    print_architecture_report(model, pep_len=9, mhc_len=180)
