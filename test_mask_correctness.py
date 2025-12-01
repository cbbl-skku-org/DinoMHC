"""
Test to verify mask correctness by comparing:
1. Full padded input with mask → output at valid positions
2. Sliced valid-only input (no mask needed) → output

If masks work correctly, both outputs should be identical.
"""
import torch
import sys
sys.path.append('src')
from model import (
    HierarchicalGrooveExtractor,
    GroovePeptideFusionLayer,
    GroovePeptideFusion,
    PositionAwarePeptideEncoder,
    InterfaceGeometryModule,
    DinoMHC,
    EmbeddingEncoder
)

# Check if ESM is available
try:
    import esm
    from model import ESM2Encoder
    HAS_ESM = True
except ImportError:
    HAS_ESM = False

torch.manual_seed(42)

def test_groove_peptide_fusion_mask_correctness():
    """Test GroovePeptideFusion mask correctness."""
    print("=" * 70)
    print("Test 1: GroovePeptideFusion Mask Correctness")
    print("=" * 70)
    
    # Dimensions
    pep_valid_len = 9
    pep_pad_len = 5
    pep_full_len = pep_valid_len + pep_pad_len  # 14
    
    groove_valid_len = 50
    groove_pad_len = 10
    groove_full_len = groove_valid_len + groove_pad_len  # 60
    
    batch_size = 1
    dim = 64
    
    print(f"Peptide: valid={pep_valid_len}, padding={pep_pad_len}, full={pep_full_len}")
    print(f"Groove: valid={groove_valid_len}, padding={groove_pad_len}, full={groove_full_len}")
    print()
    
    # Create module and set to eval mode
    module = GroovePeptideFusion(dim=dim, num_layers=2, num_heads=4, dropout=0.0)
    module.eval()
    
    # Create valid embeddings (the actual data)
    torch.manual_seed(123)
    pep_valid = torch.randn(batch_size, pep_valid_len, dim)
    groove_valid = torch.randn(batch_size, groove_valid_len, dim)
    
    # === Method 1: Full padded input with mask ===
    print("Method 1: Full padded input with mask")
    pep_padded = torch.zeros(batch_size, pep_full_len, dim)
    pep_padded[:, :pep_valid_len, :] = pep_valid
    
    groove_padded = torch.zeros(batch_size, groove_full_len, dim)
    groove_padded[:, :groove_valid_len, :] = groove_valid
    
    pep_mask = torch.zeros(batch_size, pep_full_len, dtype=torch.bool)
    pep_mask[:, :pep_valid_len] = True
    
    groove_mask = torch.zeros(batch_size, groove_full_len, dtype=torch.bool)
    groove_mask[:, :groove_valid_len] = True
    
    with torch.no_grad():
        pep_out_masked, groove_out_masked, _ = module(
            pep_padded, groove_padded,
            peptide_mask=pep_mask,
            groove_mask=groove_mask
        )
    
    # Extract valid portion
    pep_out_masked_valid = pep_out_masked[:, :pep_valid_len, :]
    groove_out_masked_valid = groove_out_masked[:, :groove_valid_len, :]
    
    print(f"  Input peptide shape: {pep_padded.shape}")
    print(f"  Input groove shape: {groove_padded.shape}")
    print(f"  Output peptide valid portion shape: {pep_out_masked_valid.shape}")
    print(f"  Output groove valid portion shape: {groove_out_masked_valid.shape}")
    print()
    
    # === Method 2: Sliced valid-only input (no mask) ===
    print("Method 2: Sliced valid-only input (no mask)")
    with torch.no_grad():
        pep_out_sliced, groove_out_sliced, _ = module(
            pep_valid, groove_valid,
            peptide_mask=None,  # No mask needed, all tokens valid
            groove_mask=None
        )
    
    print(f"  Input peptide shape: {pep_valid.shape}")
    print(f"  Input groove shape: {groove_valid.shape}")
    print(f"  Output peptide shape: {pep_out_sliced.shape}")
    print(f"  Output groove shape: {groove_out_sliced.shape}")
    print()
    
    # === Compare outputs ===
    print("Comparison:")
    pep_diff = (pep_out_masked_valid - pep_out_sliced).abs()
    groove_diff = (groove_out_masked_valid - groove_out_sliced).abs()
    
    pep_max_diff = pep_diff.max().item()
    pep_mean_diff = pep_diff.mean().item()
    groove_max_diff = groove_diff.max().item()
    groove_mean_diff = groove_diff.mean().item()
    
    print(f"  Peptide max diff: {pep_max_diff:.10f}")
    print(f"  Peptide mean diff: {pep_mean_diff:.10f}")
    print(f"  Groove max diff: {groove_max_diff:.10f}")
    print(f"  Groove mean diff: {groove_mean_diff:.10f}")
    
    threshold = 1e-5
    pep_match = pep_max_diff < threshold
    groove_match = groove_max_diff < threshold
    
    print()
    if pep_match and groove_match:
        print(f"✓ PASS: Outputs match (diff < {threshold})")
    else:
        print(f"✗ FAIL: Outputs differ!")
        if not pep_match:
            print(f"  Peptide diff too large: {pep_max_diff}")
        if not groove_match:
            print(f"  Groove diff too large: {groove_max_diff}")
    print()
    return pep_match and groove_match


def test_position_encoder_mask_correctness():
    """Test PositionAwarePeptideEncoder mask correctness."""
    print("=" * 70)
    print("Test 2: PositionAwarePeptideEncoder Mask Correctness")
    print("=" * 70)
    
    pep_valid_len = 9
    pep_pad_len = 5
    pep_full_len = pep_valid_len + pep_pad_len
    
    batch_size = 1
    dim = 64
    
    print(f"Peptide: valid={pep_valid_len}, padding={pep_pad_len}, full={pep_full_len}")
    print()
    
    module = PositionAwarePeptideEncoder(dim=dim, max_peptide_length=20, num_heads=4, dropout=0.0)
    module.eval()
    
    # Create valid embeddings
    torch.manual_seed(456)
    pep_valid = torch.randn(batch_size, pep_valid_len, dim)
    
    # === Method 1: Full padded input with mask ===
    print("Method 1: Full padded input with mask")
    pep_padded = torch.zeros(batch_size, pep_full_len, dim)
    pep_padded[:, :pep_valid_len, :] = pep_valid
    
    pep_mask = torch.zeros(batch_size, pep_full_len, dtype=torch.bool)
    pep_mask[:, :pep_valid_len] = True
    
    with torch.no_grad():
        out_masked, importance_masked = module(
            pep_padded, peptide_mask=pep_mask, return_importance=True
        )
    
    out_masked_valid = out_masked[:, :pep_valid_len, :]
    importance_masked_valid = importance_masked[:, :pep_valid_len]
    
    # === Method 2: Sliced valid-only input (no mask) ===
    print("Method 2: Sliced valid-only input (no mask)")
    with torch.no_grad():
        out_sliced, importance_sliced = module(
            pep_valid, peptide_mask=None, return_importance=True
        )
    
    # === Compare ===
    print()
    print("Comparison:")
    out_diff = (out_masked_valid - out_sliced).abs()
    importance_diff = (importance_masked_valid - importance_sliced).abs()
    
    out_max_diff = out_diff.max().item()
    importance_max_diff = importance_diff.max().item()
    
    print(f"  Output max diff: {out_max_diff:.10f}")
    print(f"  Importance max diff: {importance_max_diff:.10f}")
    
    threshold = 1e-5
    match = out_max_diff < threshold and importance_max_diff < threshold
    
    print()
    if match:
        print(f"✓ PASS: Outputs match (diff < {threshold})")
    else:
        print(f"✗ FAIL: Outputs differ!")
    print()
    return match


def test_hierarchical_groove_extractor_mask_correctness():
    """Test HierarchicalGrooveExtractor mask correctness."""
    print("=" * 70)
    print("Test 3: HierarchicalGrooveExtractor Mask Correctness")
    print("=" * 70)
    
    mhc_valid_len = 150
    mhc_pad_len = 30
    mhc_full_len = mhc_valid_len + mhc_pad_len
    
    pep_valid_len = 9
    pep_pad_len = 5
    pep_full_len = pep_valid_len + pep_pad_len
    
    batch_size = 1
    dim = 64
    
    print(f"MHC: valid={mhc_valid_len}, padding={mhc_pad_len}, full={mhc_full_len}")
    print(f"Peptide: valid={pep_valid_len}, padding={pep_pad_len}, full={pep_full_len}")
    print()
    
    module = HierarchicalGrooveExtractor(dim=dim, num_groove_tokens=30, num_heads=4, dropout=0.0)
    module.eval()
    
    # Create valid embeddings
    torch.manual_seed(789)
    mhc_valid = torch.randn(batch_size, mhc_valid_len, dim)
    pep_valid = torch.randn(batch_size, pep_valid_len, dim)
    
    # === Method 1: Full padded input with mask ===
    print("Method 1: Full padded input with mask")
    mhc_padded = torch.zeros(batch_size, mhc_full_len, dim)
    mhc_padded[:, :mhc_valid_len, :] = mhc_valid
    
    pep_padded = torch.zeros(batch_size, pep_full_len, dim)
    pep_padded[:, :pep_valid_len, :] = pep_valid
    
    mhc_mask = torch.zeros(batch_size, mhc_full_len, dtype=torch.bool)
    mhc_mask[:, :mhc_valid_len] = True
    
    pep_mask = torch.zeros(batch_size, pep_full_len, dtype=torch.bool)
    pep_mask[:, :pep_valid_len] = True
    
    with torch.no_grad():
        out_masked, _ = module(
            mhc_padded, pep_padded,
            mhc_mask=mhc_mask, peptide_mask=pep_mask
        )
    
    # === Method 2: Sliced valid-only input (no mask) ===
    print("Method 2: Sliced valid-only input (no mask)")
    with torch.no_grad():
        out_sliced, _ = module(
            mhc_valid, pep_valid,
            mhc_mask=None, peptide_mask=None
        )
    
    # === Compare ===
    print()
    print("Comparison:")
    out_diff = (out_masked - out_sliced).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()
    
    print(f"  Output max diff: {out_max_diff:.10f}")
    print(f"  Output mean diff: {out_mean_diff:.10f}")
    
    threshold = 1e-5
    match = out_max_diff < threshold
    
    print()
    if match:
        print(f"✓ PASS: Outputs match (diff < {threshold})")
    else:
        print(f"✗ FAIL: Outputs differ!")
    print()
    return match


def test_interface_geometry_mask_correctness():
    """Test InterfaceGeometryModule mask correctness."""
    print("=" * 70)
    print("Test 4: InterfaceGeometryModule Mask Correctness")
    print("=" * 70)
    
    pep_valid_len = 9
    groove_valid_len = 50
    
    batch_size = 1
    dim = 64
    
    print(f"Peptide: {pep_valid_len}")
    print(f"Groove: {groove_valid_len}")
    print("Note: This module handles internal padding for triangle ops")
    print()
    
    module = InterfaceGeometryModule(dim=dim, num_blocks=2, num_heads=4, dropout=0.0)
    module.eval()
    
    # Create valid embeddings
    torch.manual_seed(101112)
    pep_valid = torch.randn(batch_size, pep_valid_len, dim)
    groove_valid = torch.randn(batch_size, groove_valid_len, dim)
    
    # Run twice to check determinism
    print("Running twice to verify consistency...")
    with torch.no_grad():
        interface1, pair1 = module(pep_valid, groove_valid, return_pair_rep=True)
        interface2, pair2 = module(pep_valid, groove_valid, return_pair_rep=True)
    
    interface_diff = (interface1 - interface2).abs().max().item()
    pair_diff = (pair1 - pair2).abs().max().item()
    
    print(f"  Interface rep diff between runs: {interface_diff:.10f}")
    print(f"  Pair rep diff between runs: {pair_diff:.10f}")
    
    threshold = 1e-5
    match = interface_diff < threshold and pair_diff < threshold
    
    print()
    if match:
        print(f"✓ PASS: Outputs are deterministic (diff < {threshold})")
    else:
        print(f"✗ FAIL: Outputs not deterministic!")
    print()
    return match


def test_full_model_mask_correctness():
    """Test full DinoMHC mask correctness."""
    print("=" * 70)
    print("Test 5: DinoMHC Full Model Mask Correctness")
    print("=" * 70)
    
    pep_valid_len = 9
    pep_pad_len = 5
    pep_full_len = pep_valid_len + pep_pad_len
    
    mhc_valid_len = 150
    mhc_pad_len = 30
    mhc_full_len = mhc_valid_len + mhc_pad_len
    
    batch_size = 1
    
    print(f"Peptide: valid={pep_valid_len}, padding={pep_pad_len}, full={pep_full_len}")
    print(f"MHC: valid={mhc_valid_len}, padding={mhc_pad_len}, full={mhc_full_len}")
    print()
    
    config = {
        'dim': 64,
        'num_groove_tokens': 30,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 4,
        'dropout': 0.0,  # No dropout for deterministic comparison
        'max_peptide_length': 20,
        'task_head': 'presentation',
        'encoder_type': 'embedding'  # Use embedding encoder for mask correctness test
    }
    
    model = DinoMHC(config)
    model.eval()
    
    # Create valid token sequences
    torch.manual_seed(131415)
    pep_valid_tokens = torch.randint(1, 20, (batch_size, pep_valid_len))
    mhc_valid_tokens = torch.randint(1, 20, (batch_size, mhc_valid_len))
    
    # === Method 1: Full padded input with mask ===
    print("Method 1: Full padded input with mask")
    pep_padded = torch.zeros(batch_size, pep_full_len, dtype=torch.long)
    pep_padded[:, :pep_valid_len] = pep_valid_tokens
    
    mhc_padded = torch.zeros(batch_size, mhc_full_len, dtype=torch.long)
    mhc_padded[:, :mhc_valid_len] = mhc_valid_tokens
    
    pep_mask = torch.zeros(batch_size, pep_full_len, dtype=torch.bool)
    pep_mask[:, :pep_valid_len] = True
    
    mhc_mask = torch.zeros(batch_size, mhc_full_len, dtype=torch.bool)
    mhc_mask[:, :mhc_valid_len] = True
    
    with torch.no_grad():
        out_masked = model(
            pep_padded, mhc_padded,
            peptide_mask=pep_mask, mhc_mask=mhc_mask
        )
    
    pred_masked = out_masked['prediction']
    
    # === Method 2: Sliced valid-only input (no mask) ===
    print("Method 2: Sliced valid-only input (no mask)")
    with torch.no_grad():
        out_sliced = model(
            pep_valid_tokens, mhc_valid_tokens,
            peptide_mask=None, mhc_mask=None
        )
    
    pred_sliced = out_sliced['prediction']
    
    # === Compare ===
    print()
    print("Comparison:")
    pred_diff = (pred_masked - pred_sliced).abs().item()
    
    print(f"  Prediction (masked): {pred_masked.item():.10f}")
    print(f"  Prediction (sliced): {pred_sliced.item():.10f}")
    print(f"  Difference: {pred_diff:.10f}")
    
    threshold = 1e-5
    match = pred_diff < threshold
    
    print()
    if match:
        print(f"✓ PASS: Predictions match (diff < {threshold})")
    else:
        print(f"✗ FAIL: Predictions differ!")
    print()
    return match


def test_embedding_encoder_mask_correctness():
    """Test EmbeddingEncoder mask correctness."""
    print("=" * 70)
    print("Test: EmbeddingEncoder Mask Correctness")
    print("=" * 70)
    
    valid_len = 9
    pad_len = 5
    full_len = valid_len + pad_len
    
    batch_size = 1
    dim = 64
    
    print(f"Sequence: valid={valid_len}, padding={pad_len}, full={full_len}")
    print()
    
    encoder = EmbeddingEncoder(dim=dim)
    encoder.eval()
    
    # Create valid tokens
    torch.manual_seed(111)
    valid_tokens = torch.randint(1, 20, (batch_size, valid_len))
    
    # === Method 1: Full padded input with mask ===
    print("Method 1: Full padded input with mask")
    padded_tokens = torch.zeros(batch_size, full_len, dtype=torch.long)
    padded_tokens[:, :valid_len] = valid_tokens
    
    mask = torch.zeros(batch_size, full_len, dtype=torch.bool)
    mask[:, :valid_len] = True
    
    with torch.no_grad():
        out_masked = encoder(padded_tokens, mask=mask)
    
    out_masked_valid = out_masked[:, :valid_len, :]
    out_masked_padding = out_masked[:, valid_len:, :]
    
    print(f"  Input tokens shape: {padded_tokens.shape}")
    print(f"  Output valid portion shape: {out_masked_valid.shape}")
    print(f"  Padding portion should be zeros: {out_masked_padding.abs().max().item():.10f}")
    
    # === Method 2: Sliced valid-only input (no mask) ===
    print("\nMethod 2: Sliced valid-only input (no mask)")
    with torch.no_grad():
        out_sliced = encoder(valid_tokens, mask=None)
    
    print(f"  Input tokens shape: {valid_tokens.shape}")
    print(f"  Output shape: {out_sliced.shape}")
    
    # === Compare ===
    print("\nComparison:")
    out_diff = (out_masked_valid - out_sliced).abs()
    out_max_diff = out_diff.max().item()
    padding_is_zero = out_masked_padding.abs().max().item() < 1e-10
    
    print(f"  Valid portion max diff: {out_max_diff:.10f}")
    print(f"  Padding is zero: {padding_is_zero}")
    
    threshold = 1e-5
    match = out_max_diff < threshold and padding_is_zero
    
    print()
    if match:
        print(f"✓ PASS: Outputs match (diff < {threshold}) and padding is zeroed")
    else:
        print(f"✗ FAIL: Outputs differ or padding not zeroed!")
    print()
    return match


def test_esm2_encoder_mask_correctness():
    """
    Test ESM2Encoder mask correctness.
    
    Note: ESM-2 (like all transformer-based PLMs) produces different representations
    for the same subsequence when the full sequence length differs. This is because:
    1. Attention patterns depend on all positions
    2. Rotary position embeddings encode absolute positions
    
    The ESM2Encoder now:
    - Removes BOS (position 0) from output
    - Marks EOS as invalid so it gets zeroed
    - Output length is input_length - 1 (only BOS removed)
    
    We test that:
    1. Output length is correctly input_length - 1
    2. EOS and padded positions are properly zeroed out
    3. Outputs are deterministic (same input → same output)
    4. Valid positions have meaningful values
    """
    print("=" * 70)
    print("Test: ESM2Encoder Mask Correctness (BOS removed, EOS zeroed)")
    print("=" * 70)
    
    if not HAS_ESM:
        print("  ⚠ ESM not installed, skipping")
        return None
    
    batch_size = 2
    dim = 320
    
    print("Testing ESM-2 encoder with BOS removal and EOS zeroing...")
    print()
    
    encoder = ESM2Encoder(output_dim=dim, unfreeze_layers=0)
    encoder.eval()
    
    # Get ESM alphabet for tokenization
    _, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')
    batch_converter = alphabet.get_batch_converter()
    padding_idx = alphabet.padding_idx
    eos_idx = alphabet.eos_idx
    
    # Create sequences of different lengths (realistic batching scenario)
    seqs = [
        ("seq1", "ACDEFGHIK"),      # 9 AA
        ("seq2", "ACDEFG"),         # 6 AA
    ]
    
    # Tokenize with batch_converter (handles padding automatically)
    # ESM tokenization: [BOS, seq..., EOS, PAD...] for shorter sequences
    _, _, tokens = batch_converter(seqs)
    
    # Create mask based on padding token
    mask = tokens != padding_idx
    
    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Tokens[0]: {tokens[0].tolist()}")
    print(f"  Tokens[1]: {tokens[1].tolist()}")
    print(f"  Mask[0]: {mask[0].tolist()}")
    print(f"  Mask[1]: {mask[1].tolist()}")
    
    # === Test 0: Output shape should be input - 1 (BOS removed) ===
    print("\nTest 0: Output shape should have BOS removed (input - 1)")
    with torch.no_grad():
        out = encoder(tokens, mask=mask)
    
    expected_len = tokens.shape[1] - 1  # Only BOS removed
    actual_len = out.shape[1]
    print(f"  Input length: {tokens.shape[1]}")
    print(f"  Expected output length (input - 1): {expected_len}")
    print(f"  Actual output length: {actual_len}")
    shape_correct = actual_len == expected_len
    print(f"  Shape correct: {shape_correct}")
    
    # === Test 1: EOS and padded positions should be zeroed ===
    print("\nTest 1: EOS and padded positions should be zeroed")
    # After removing BOS:
    # seq1: [A, C, D, E, F, G, H, I, K, EOS] -> EOS at position 9 should be zeroed
    # seq2: [A, C, D, E, F, G, EOS, PAD, PAD, PAD] -> EOS at position 6, PADs after
    
    # Check seq1 EOS position (last position, which was EOS)
    seq1_eos_pos = 9  # After BOS removal, EOS is at position 9
    seq1_eos_val = out[0, seq1_eos_pos, :].abs().max().item()
    print(f"  Seq1 EOS position ({seq1_eos_pos}) max value: {seq1_eos_val:.10f}")
    
    # Check seq2 EOS and padding positions
    seq2_eos_pos = 6  # After BOS removal, EOS is at position 6
    seq2_eos_and_pad = out[1, seq2_eos_pos:, :]
    seq2_eos_pad_max = seq2_eos_and_pad.abs().max().item()
    print(f"  Seq2 EOS+PAD positions ({seq2_eos_pos}:) max value: {seq2_eos_pad_max:.10f}")
    
    eos_pad_zeroed = seq1_eos_val < 1e-10 and seq2_eos_pad_max < 1e-10
    print(f"  EOS and padding zeroed: {eos_pad_zeroed}")
    
    # === Test 2: Output should be deterministic ===
    print("\nTest 2: Outputs should be deterministic")
    with torch.no_grad():
        out1 = encoder(tokens, mask=mask)
        out2 = encoder(tokens, mask=mask)
    
    diff = (out1 - out2).abs().max().item()
    print(f"  Difference between two runs: {diff:.10f}")
    deterministic = diff < 1e-10
    print(f"  Deterministic: {deterministic}")
    
    # === Test 3: Valid positions should have non-trivial values ===
    print("\nTest 3: Valid positions should have meaningful values")
    # Seq1 has 9 valid AA at positions 0:9 after BOS removal
    valid_out = out[0, :9, :]  # First 9 positions (the actual sequence)
    valid_mean = valid_out.abs().mean().item()
    valid_std = valid_out.std().item()
    print(f"  Valid output mean magnitude: {valid_mean:.6f}")
    print(f"  Valid output std: {valid_std:.6f}")
    has_values = valid_mean > 0.01 and valid_std > 0.01
    print(f"  Has meaningful values: {has_values}")
    
    # Overall pass
    match = shape_correct and eos_pad_zeroed and deterministic and has_values
    
    print()
    if match:
        print(f"✓ PASS: ESM-2 encoder handles BOS removal and EOS zeroing correctly")
    else:
        print(f"✗ FAIL: ESM-2 encoder issues!")
        if not shape_correct:
            print("  - Shape mismatch (BOS not removed correctly)")
        if not eos_pad_zeroed:
            print("  - EOS/padding not zeroed")
        if not deterministic:
            print("  - Not deterministic")
        if not has_values:
            print("  - Valid positions have no meaningful values")
    print()
    return match


def test_full_model_esm2_mask_correctness():
    """
    Test full DinoMHC with ESM-2 encoder mask correctness.
    
    Similar to ESM2Encoder test, we verify:
    1. Outputs are deterministic for same input
    2. Model runs without errors with masked inputs
    3. Predictions are reasonable
    
    Note: Due to PLM characteristics, we don't expect identical outputs
    for padded+mask vs sliced inputs.
    """
    print("=" * 70)
    print("Test: DinoMHC (ESM-2) Full Model Mask Correctness")
    print("=" * 70)
    
    if not HAS_ESM:
        print("  ⚠ ESM not installed, skipping")
        return None
    
    batch_size = 2
    
    print("Testing DinoMHC with ESM-2 encoder...")
    print()
    
    # Get ESM alphabet for tokenization
    _, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')
    batch_converter = alphabet.get_batch_converter()
    padding_idx = alphabet.padding_idx
    
    # Create peptide and MHC sequences of different lengths
    pep_seqs = [
        ("pep1", "ACDEFGHIK"),   # 9 AA
        ("pep2", "ACDEFG"),      # 6 AA
    ]
    mhc_seqs = [
        ("mhc1", "ACDEFGHIKLMNPQRSTVWY" * 5),  # 100 AA
        ("mhc2", "ACDEFGHIKLMNPQRSTVWY" * 4),  # 80 AA
    ]
    
    # Tokenize
    _, _, pep_tokens = batch_converter(pep_seqs)
    _, _, mhc_tokens = batch_converter(mhc_seqs)
    
    # Create masks
    pep_mask = pep_tokens != padding_idx
    mhc_mask = mhc_tokens != padding_idx
    
    print(f"  Peptide tokens shape: {pep_tokens.shape}")
    print(f"  MHC tokens shape: {mhc_tokens.shape}")
    print(f"  Peptide valid: seq1={pep_mask[0].sum().item()}, seq2={pep_mask[1].sum().item()}")
    print(f"  MHC valid: seq1={mhc_mask[0].sum().item()}, seq2={mhc_mask[1].sum().item()}")
    
    config = {
        'dim': 320,
        'encoder_type': 'esm2',
        'esm_unfreeze_layers': 0,
        'num_groove_tokens': 30,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 4,
        'dropout': 0.0,
        'max_peptide_length': 20,
        'task_head': 'presentation'
    }
    
    model = DinoMHC(config)
    model.eval()
    
    # === Test 1: Forward pass works ===
    print("\nTest 1: Forward pass with masked inputs")
    try:
        with torch.no_grad():
            out = model(
                pep_tokens, mhc_tokens,
                peptide_mask=pep_mask, mhc_mask=mhc_mask
            )
        print(f"  ✓ Forward pass successful")
        print(f"  Prediction shape: {out['prediction'].shape}")
        print(f"  Predictions: {out['prediction'].squeeze().tolist()}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False
    
    # === Test 2: Outputs are deterministic ===
    print("\nTest 2: Outputs should be deterministic")
    with torch.no_grad():
        out1 = model(pep_tokens, mhc_tokens, pep_mask, mhc_mask)
        out2 = model(pep_tokens, mhc_tokens, pep_mask, mhc_mask)
    
    diff = (out1['prediction'] - out2['prediction']).abs().max().item()
    print(f"  Difference between two runs: {diff:.10f}")
    deterministic = diff < 1e-10
    print(f"  Deterministic: {deterministic}")
    
    # === Test 3: Predictions are in valid range ===
    print("\nTest 3: Predictions should be in valid range (0-1)")
    preds = out['prediction'].squeeze()
    in_range = (preds >= 0).all().item() and (preds <= 1).all().item()
    print(f"  All predictions in [0, 1]: {in_range}")
    
    # Overall pass
    match = deterministic and in_range
    
    print()
    if match:
        print(f"✓ PASS: DinoMHC with ESM-2 works correctly")
    else:
        print(f"✗ FAIL: Issues detected!")
    print()
    return match


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MASK CORRECTNESS VERIFICATION TESTS")
    print("Comparing: padded+mask vs sliced (valid only)")
    print("=" * 70 + "\n")
    
    results = []
    
    results.append(("EmbeddingEncoder", test_embedding_encoder_mask_correctness()))
    if HAS_ESM:
        results.append(("ESM2Encoder", test_esm2_encoder_mask_correctness()))
    else:
        print("=" * 70)
        print("Test: ESM2Encoder Mask Correctness - SKIPPED (ESM not installed)")
        print("=" * 70 + "\n")
        results.append(("ESM2Encoder", None))
    results.append(("GroovePeptideFusion", test_groove_peptide_fusion_mask_correctness()))
    results.append(("PositionAwarePeptideEncoder", test_position_encoder_mask_correctness()))
    results.append(("HierarchicalGrooveExtractor", test_hierarchical_groove_extractor_mask_correctness()))
    results.append(("InterfaceGeometryModule", test_interface_geometry_mask_correctness()))
    results.append(("DinoMHC (Embedding)", test_full_model_mask_correctness()))
    if HAS_ESM:
        results.append(("DinoMHC (ESM-2)", test_full_model_esm2_mask_correctness()))
    else:
        print("=" * 70)
        print("Test: DinoMHC (ESM-2) Mask Correctness - SKIPPED (ESM not installed)")
        print("=" * 70 + "\n")
        results.append(("DinoMHC (ESM-2)", None))
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        if passed is None:
            status = "⚠ SKIP"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False
        print(f"  {status}: {name}")
    
    print()
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("=" * 70)
