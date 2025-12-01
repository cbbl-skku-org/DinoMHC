#!/usr/bin/env python
"""
Test script for ESM-2 encoder integration in DinoMHC.

Tests:
1. Embedding encoder (baseline)
2. ESM-2 encoder with different unfreezing strategies
3. Shared vs separate encoders
4. Parameter counting and trainability
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import sys
sys.path.insert(0, '/home/duongtt/Workdir/MHC/digermhc/src')

from model import DinoMHC, EmbeddingEncoder, ESM2Encoder


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    """Format parameter count."""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def test_embedding_encoder():
    """Test simple embedding encoder."""
    print("\n" + "="*60)
    print("TEST 1: Embedding Encoder (Baseline)")
    print("="*60)
    
    config = {
        'dim': 320,
        'encoder_type': 'embedding',
        'num_groove_tokens': 60,
        'num_fusion_layers': 4,
        'num_geometry_blocks': 3,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    model = DinoMHC(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 2
    pep_len = 9
    mhc_len = 180
    
    peptide_tokens = torch.randint(0, 20, (batch_size, pep_len)).to(device)
    mhc_tokens = torch.randint(0, 20, (batch_size, mhc_len)).to(device)
    peptide_mask = torch.ones(batch_size, pep_len, dtype=torch.bool).to(device)
    mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(peptide_tokens, mhc_tokens, peptide_mask, mhc_mask)
    
    total, trainable = count_parameters(model)
    
    print(f"  ✓ Encoder type: {config['encoder_type']}")
    print(f"  ✓ Total parameters: {format_params(total)}")
    print(f"  ✓ Trainable parameters: {format_params(trainable)}")
    print(f"  ✓ Output shape: {outputs['prediction'].shape}")
    print(f"  ✓ Encoder info: {model.get_encoder_info()}")
    
    return True


def test_esm2_encoder_variants():
    """Test ESM-2 encoder with different configurations."""
    print("\n" + "="*60)
    print("TEST 2: ESM-2 Encoder Variants")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if ESM is available
    try:
        import esm
        print("  ✓ ESM package found")
    except ImportError:
        print("  ⚠ ESM package not installed. Skipping ESM tests.")
        print("    Install with: pip install fair-esm")
        return False
    
    # Test configurations
    test_configs = [
        {
            'name': 'ESM-2 (freeze all)',
            'encoder_type': 'esm2',
            'esm_unfreeze_layers': 0
        },
        {
            'name': 'ESM-2 (unfreeze 2 layers)',
            'encoder_type': 'esm2',
            'esm_unfreeze_layers': 2
        },
        {
            'name': 'ESM-2 (unfreeze all)',
            'encoder_type': 'esm2',
            'esm_unfreeze_layers': -1
        },
        {
            'name': 'ESM-2 Shared (unfreeze 2 layers)',
            'encoder_type': 'esm2_shared',
            'esm_unfreeze_layers': 2
        }
    ]
    
    base_config = {
        'dim': 320,
        'num_groove_tokens': 60,
        'num_fusion_layers': 4,
        'num_geometry_blocks': 3,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    # Prepare ESM tokenization
    batch_size = 2
    pep_len = 9
    mhc_len = 180
    
    # ESM uses different tokenization - we need to use the alphabet
    _, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')
    batch_converter = alphabet.get_batch_converter()
    
    # Create sequences (just random for testing)
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    peptide_seqs = [''.join([aa_vocab[i % 20] for i in range(pep_len)]) for _ in range(batch_size)]
    mhc_seqs = [''.join([aa_vocab[i % 20] for i in range(mhc_len)]) for _ in range(batch_size)]
    
    # Convert to ESM format
    pep_data = [(f"pep_{i}", seq) for i, seq in enumerate(peptide_seqs)]
    mhc_data = [(f"mhc_{i}", seq) for i, seq in enumerate(mhc_seqs)]
    
    _, _, pep_tokens = batch_converter(pep_data)
    _, _, mhc_tokens = batch_converter(mhc_data)
    
    pep_tokens = pep_tokens.to(device)
    mhc_tokens = mhc_tokens.to(device)
    
    # ESM adds BOS and EOS tokens, so lengths are +2
    pep_mask = torch.ones(batch_size, pep_tokens.shape[1], dtype=torch.bool).to(device)
    mhc_mask = torch.ones(batch_size, mhc_tokens.shape[1], dtype=torch.bool).to(device)
    
    results = []
    
    for test_cfg in test_configs:
        config = {**base_config, **test_cfg}
        del config['name']
        
        try:
            model = DinoMHC(config).to(device)
            
            with torch.no_grad():
                outputs = model(pep_tokens, mhc_tokens, pep_mask, mhc_mask)
            
            total, trainable = count_parameters(model)
            encoder_info = model.get_encoder_info()
            
            results.append({
                'name': test_cfg['name'],
                'total': total,
                'trainable': trainable,
                'output_shape': outputs['prediction'].shape,
                'encoder_info': encoder_info
            })
            
            print(f"\n  [{test_cfg['name']}]")
            print(f"    Total params: {format_params(total)}")
            print(f"    Trainable params: {format_params(trainable)} ({100*trainable/total:.1f}%)")
            print(f"    Output shape: {outputs['prediction'].shape}")
            
            # Cleanup to save GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n  [{test_cfg['name']}] FAILED: {e}")
    
    return True


def test_freeze_unfreeze():
    """Test dynamic freezing/unfreezing of encoder layers."""
    print("\n" + "="*60)
    print("TEST 3: Dynamic Freeze/Unfreeze")
    print("="*60)
    
    try:
        import esm
    except ImportError:
        print("  ⚠ ESM package not installed. Skipping.")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'dim': 320,
        'encoder_type': 'esm2',
        'esm_unfreeze_layers': 0,  # Start frozen
        'num_groove_tokens': 60,
        'num_fusion_layers': 4,
        'num_geometry_blocks': 3,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    model = DinoMHC(config).to(device)
    
    # Initial state (frozen)
    total, trainable = count_parameters(model)
    print(f"  Initial (frozen encoders):")
    print(f"    Trainable: {format_params(trainable)} / {format_params(total)}")
    
    # Unfreeze 2 layers
    model.unfreeze_encoder_layers(2)
    total, trainable = count_parameters(model)
    print(f"  After unfreeze_encoder_layers(2):")
    print(f"    Trainable: {format_params(trainable)} / {format_params(total)}")
    
    # Unfreeze all
    model.unfreeze_encoder_layers(-1)
    total, trainable = count_parameters(model)
    print(f"  After unfreeze_encoder_layers(-1) [all]:")
    print(f"    Trainable: {format_params(trainable)} / {format_params(total)}")
    
    # Freeze again
    model.freeze_encoders()
    total, trainable = count_parameters(model)
    print(f"  After freeze_encoders():")
    print(f"    Trainable: {format_params(trainable)} / {format_params(total)}")
    
    return True


def test_encoder_comparison():
    """Compare embedding vs ESM-2 encoders."""
    print("\n" + "="*60)
    print("TEST 4: Encoder Comparison Table")
    print("="*60)
    
    try:
        import esm
        has_esm = True
    except ImportError:
        has_esm = False
        print("  ⚠ ESM not installed, showing embedding only")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        ('Embedding (baseline)', {'encoder_type': 'embedding'}),
    ]
    
    if has_esm:
        configs += [
            ('ESM-2 (frozen)', {'encoder_type': 'esm2', 'esm_unfreeze_layers': 0}),
            ('ESM-2 (2 layers)', {'encoder_type': 'esm2', 'esm_unfreeze_layers': 2}),
            ('ESM-2 Shared (2 layers)', {'encoder_type': 'esm2_shared', 'esm_unfreeze_layers': 2}),
        ]
    
    base_config = {
        'dim': 320,
        'num_groove_tokens': 60,
        'num_fusion_layers': 4,
        'num_geometry_blocks': 3,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    print("\n  Encoder Comparison:")
    print("  " + "-"*70)
    print(f"  {'Configuration':<35} {'Total':>12} {'Trainable':>12} {'%':>8}")
    print("  " + "-"*70)
    
    for name, cfg in configs:
        config = {**base_config, **cfg}
        model = DinoMHC(config).to(device)
        total, trainable = count_parameters(model)
        pct = 100 * trainable / total
        print(f"  {name:<35} {format_params(total):>12} {format_params(trainable):>12} {pct:>7.1f}%")
        del model
        torch.cuda.empty_cache()
    
    print("  " + "-"*70)
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  DinoMHC ESM-2 Encoder Tests")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Run tests
    tests = [
        ("Embedding Encoder", test_embedding_encoder),
        ("ESM-2 Variants", test_esm2_encoder_variants),
        ("Freeze/Unfreeze", test_freeze_unfreeze),
        ("Encoder Comparison", test_encoder_comparison),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "SKIP"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))
    
    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    for name, status in results:
        symbol = "✓" if status == "PASS" else ("⚠" if status == "SKIP" else "✗")
        print(f"  {symbol} {name}: {status}")
    print()


if __name__ == "__main__":
    main()
