"""
Test script for HierarchicalGrooveExtractor with padding handling.
"""
import torch
import sys
sys.path.append('src')
from model import HierarchicalGrooveExtractor


def test_basic_forward():
    """Test basic forward pass without padding."""
    print("=" * 60)
    print("Test 1: Basic Forward Pass (No Padding)")
    print("=" * 60)

    batch_size = 2
    mhc_len = 180
    pep_len = 12
    dim = 128

    # Create model
    extractor = HierarchicalGrooveExtractor(
        dim=dim,
        num_groove_tokens=60,
        num_regions=3,
        num_heads=8
    )

    # Create dummy data
    mhc_emb = torch.randn(batch_size, mhc_len, dim)
    peptide_emb = torch.randn(batch_size, pep_len, dim)

    # Forward pass
    groove_features, attn_info = extractor(
        mhc_emb=mhc_emb,
        peptide_emb=peptide_emb,
        return_attention=True
    )

    print(f"✓ MHC embedding shape: {mhc_emb.shape}")
    print(f"✓ Peptide embedding shape: {peptide_emb.shape}")
    print(f"✓ Groove features shape: {groove_features.shape}")
    print(f"✓ Region attention shape: {attn_info['region_attention'].shape}")
    print(f"✓ Groove attention shape: {attn_info['groove_attention'].shape}")
    print()


def test_with_padding():
    """Test forward pass with padding masks."""
    print("=" * 60)
    print("Test 2: Forward Pass with Padding")
    print("=" * 60)

    batch_size = 3
    mhc_len = 200
    pep_len = 15
    dim = 128

    # Create model
    extractor = HierarchicalGrooveExtractor(
        dim=dim,
        num_groove_tokens=60,
        num_regions=3,
        num_heads=8
    )

    # Create dummy data with different sequence lengths
    mhc_emb = torch.randn(batch_size, mhc_len, dim)
    peptide_emb = torch.randn(batch_size, pep_len, dim)

    # Create masks with different valid lengths
    mhc_lengths = [180, 150, 120]  # Different MHC lengths
    pep_lengths = [12, 9, 8]       # Different peptide lengths

    mhc_mask = torch.zeros(batch_size, mhc_len, dtype=torch.bool)
    peptide_mask = torch.zeros(batch_size, pep_len, dtype=torch.bool)

    for i in range(batch_size):
        mhc_mask[i, :mhc_lengths[i]] = True
        peptide_mask[i, :pep_lengths[i]] = True

    print(f"Sample {0}: MHC length={mhc_lengths[0]}, Peptide length={pep_lengths[0]}")
    print(f"Sample {1}: MHC length={mhc_lengths[1]}, Peptide length={pep_lengths[1]}")
    print(f"Sample {2}: MHC length={mhc_lengths[2]}, Peptide length={pep_lengths[2]}")
    print()

    # Forward pass with masks
    groove_features, attn_info = extractor(
        mhc_emb=mhc_emb,
        peptide_emb=peptide_emb,
        mhc_mask=mhc_mask,
        peptide_mask=peptide_mask,
        return_attention=True
    )

    print(f"✓ Groove features shape: {groove_features.shape}")
    print(f"✓ No NaN values in output: {not torch.isnan(groove_features).any()}")
    print()


def test_masked_pooling():
    """Test that masked pooling works correctly."""
    print("=" * 60)
    print("Test 3: Masked Mean Pooling Verification")
    print("=" * 60)

    batch_size = 2
    mhc_len = 100
    pep_len = 10
    dim = 128

    extractor = HierarchicalGrooveExtractor(dim=dim)

    # Create peptide embeddings with known values
    peptide_emb = torch.ones(batch_size, pep_len, dim)
    peptide_emb[0, :5] = 2.0   # First sample: 5 tokens with value 2.0
    peptide_emb[1, :8] = 3.0   # Second sample: 8 tokens with value 3.0

    # Create masks
    peptide_mask = torch.zeros(batch_size, pep_len, dtype=torch.bool)
    peptide_mask[0, :5] = True  # Only first 5 tokens are valid
    peptide_mask[1, :8] = True  # Only first 8 tokens are valid

    mhc_emb = torch.randn(batch_size, mhc_len, dim)
    mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool)

    # Forward pass
    groove_features_masked, _ = extractor(
        mhc_emb=mhc_emb,
        peptide_emb=peptide_emb,
        mhc_mask=mhc_mask,
        peptide_mask=peptide_mask
    )

    # Forward pass without mask (should give different results)
    groove_features_unmasked, _ = extractor(
        mhc_emb=mhc_emb,
        peptide_emb=peptide_emb,
        mhc_mask=mhc_mask,
        peptide_mask=None  # No peptide mask
    )

    # Results should be different because of different mean pooling
    difference = (groove_features_masked - groove_features_unmasked).abs().mean()

    print(f"✓ Peptide mask sample 0: {peptide_mask[0]}")
    print(f"✓ Peptide mask sample 1: {peptide_mask[1]}")
    print(f"✓ Mean difference between masked and unmasked: {difference:.4f}")
    print(f"✓ Masked pooling has effect: {difference > 0.01}")
    print()


def test_attention_weights():
    """Verify attention doesn't attend to padding positions."""
    print("=" * 60)
    print("Test 4: Attention Weight Distribution")
    print("=" * 60)

    batch_size = 2
    mhc_len = 50
    pep_len = 10
    dim = 64

    extractor = HierarchicalGrooveExtractor(
        dim=dim,
        num_groove_tokens=20,
        num_regions=2,
        num_heads=4
    )

    mhc_emb = torch.randn(batch_size, mhc_len, dim)
    peptide_emb = torch.randn(batch_size, pep_len, dim)

    # First sample: full sequence
    # Second sample: half sequence with padding
    mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool)
    mhc_mask[1, 25:] = False  # Second sample has padding after position 25

    # Forward with attention weights
    _, attn_info = extractor(
        mhc_emb=mhc_emb,
        peptide_emb=peptide_emb,
        mhc_mask=mhc_mask,
        return_attention=True
    )

    region_attn = attn_info['region_attention']  # [batch, num_regions, mhc_len]
    groove_attn = attn_info['groove_attention']  # [batch, num_groove_tokens, mhc_len]

    print(f"✓ Region attention shape: {region_attn.shape}")
    print(f"✓ Groove attention shape: {groove_attn.shape}")
    print()

    # Check attention on padded positions
    # For sample 1 (second sample), positions 25: should have near-zero attention
    region_attn_on_padding = region_attn[1, :, 25:].abs().mean()
    groove_attn_on_padding = groove_attn[1, :, 25:].abs().mean()

    print(f"Sample 1 (no padding): Region attention sum = {region_attn[0].sum():.4f}")
    print(f"Sample 2 (with padding): Region attention sum = {region_attn[1].sum():.4f}")
    print(f"Sample 2 attention on padding positions (should be ~0): {region_attn_on_padding:.6f}")
    print(f"Sample 2 groove attention on padding (should be ~0): {groove_attn_on_padding:.6f}")
    print()


def test_without_peptide():
    """Test forward pass without peptide embeddings."""
    print("=" * 60)
    print("Test 5: Forward Pass without Peptide Embeddings")
    print("=" * 60)

    batch_size = 2
    mhc_len = 150
    dim = 128

    extractor = HierarchicalGrooveExtractor(dim=dim)

    mhc_emb = torch.randn(batch_size, mhc_len, dim)
    mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool)
    mhc_mask[1, 100:] = False  # Second sample has padding

    # Forward without peptide
    groove_features, _ = extractor(
        mhc_emb=mhc_emb,
        mhc_mask=mhc_mask
    )

    print(f"✓ Groove features shape: {groove_features.shape}")
    print(f"✓ No NaN values: {not torch.isnan(groove_features).any()}")
    print(f"✓ Successfully processed without peptide embeddings")
    print()


def test_edge_cases():
    """Test edge cases like very short sequences."""
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)

    batch_size = 3
    mhc_len = 100
    pep_len = 20
    dim = 64

    extractor = HierarchicalGrooveExtractor(dim=dim, num_heads=4)

    mhc_emb = torch.randn(batch_size, mhc_len, dim)
    peptide_emb = torch.randn(batch_size, pep_len, dim)

    # Edge case 1: Very short sequence (length 1)
    # Edge case 2: Normal sequence
    # Edge case 3: Long sequence
    mhc_mask = torch.zeros(batch_size, mhc_len, dtype=torch.bool)
    peptide_mask = torch.zeros(batch_size, pep_len, dtype=torch.bool)

    mhc_mask[0, :1] = True      # Only 1 token
    mhc_mask[1, :50] = True     # Normal length
    mhc_mask[2, :] = True       # Full length

    peptide_mask[0, :1] = True   # Only 1 token
    peptide_mask[1, :10] = True  # Normal length
    peptide_mask[2, :] = True    # Full length

    print(f"Sample 0: MHC length=1, Peptide length=1 (extreme case)")
    print(f"Sample 1: MHC length=50, Peptide length=10 (normal)")
    print(f"Sample 2: MHC length=100, Peptide length=20 (full)")
    print()

    try:
        groove_features, _ = extractor(
            mhc_emb=mhc_emb,
            peptide_emb=peptide_emb,
            mhc_mask=mhc_mask,
            peptide_mask=peptide_mask
        )

        print(f"✓ Handled extreme case (length=1)")
        print(f"✓ Groove features shape: {groove_features.shape}")
        print(f"✓ No NaN values: {not torch.isnan(groove_features).any()}")
        print(f"✓ All edge cases passed!")

    except Exception as e:
        print(f"✗ Error in edge case: {e}")

    print()

def test_value_with_wo_padding():
    """Test that masked sequence produces same output as physically truncated sequence."""
    print("=" * 60)
    print("Test 7: Masked vs Truncated Sequence Equivalence")
    print("=" * 60)

    batch_size = 2
    mhc_len = 120
    pep_len = 15
    dim = 128

    extractor = HierarchicalGrooveExtractor(dim=dim)
    extractor.eval()  # Disable dropout for deterministic comparison

    # Create SAME data for both tests
    mhc_emb_full = torch.randn(batch_size, mhc_len, dim)
    peptide_emb_full = torch.randn(batch_size, pep_len, dim)

    # Test 1: Full sequence with masking
    mhc_mask_full = torch.ones(batch_size, mhc_len, dtype=torch.bool)
    mhc_mask_full[:, 80:] = False  # Mask positions 80-119

    peptide_mask_full = torch.ones(batch_size, pep_len, dtype=torch.bool)
    peptide_mask_full[:, 10:] = False  # Mask positions 10-14

    groove_with_padding, _ = extractor(
        mhc_emb=mhc_emb_full,
        peptide_emb=peptide_emb_full,
        mhc_mask=mhc_mask_full,
        peptide_mask=peptide_mask_full
    )

    # Test 2: Physically truncated sequence (no padding tokens)
    # Use the SAME embeddings, just truncated
    mhc_emb_truncated = mhc_emb_full[:, :80, :]
    peptide_emb_truncated = peptide_emb_full[:, :10, :]

    # No mask needed (or all True mask) since there's no padding
    groove_without_padding, _ = extractor(
        mhc_emb=mhc_emb_truncated,
        peptide_emb=peptide_emb_truncated,
        mhc_mask=None,  # No padding, so no mask needed
        peptide_mask=None
    )
    
    print(groove_with_padding.shape)
    print(groove_without_padding.shape)

    # These should be IDENTICAL (or very close due to numerical precision)
    difference = (groove_with_padding - groove_without_padding).abs().mean()

    print(f"MHC: Full with mask (length={mhc_len}, valid={mhc_mask_full[0].sum().item()})")
    print(f"MHC: Truncated (length={mhc_emb_truncated.shape[1]})")
    print(f"Peptide: Full with mask (length={pep_len}, valid={peptide_mask_full[0].sum().item()})")
    print(f"Peptide: Truncated (length={peptide_emb_truncated.shape[1]})")
    print()
    print(f"✓ Groove features with padding shape: {groove_with_padding.shape}")
    print(f"✓ Groove features without padding shape: {groove_without_padding.shape}")
    print(f"✓ Mean absolute difference: {difference:.6f}")
    print(f"✓ Outputs are equivalent (diff < 1e-5): {difference < 1e-5}")

    if difference >= 1e-5:
        print(f"⚠ Warning: Difference is larger than expected!")
        print(f"  This suggests padding masking may not be working correctly.")
    else:
        print(f"✓ Perfect! Masked sequence = Truncated sequence")
    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    print("\n" + "=" * 60)
    print("HIERARCHICAL GROOVE EXTRACTOR TEST SUITE")
    print("=" * 60 + "\n")

    # Run all tests
    test_basic_forward()
    test_with_padding()
    test_masked_pooling()
    test_attention_weights()
    test_without_peptide()
    test_edge_cases()
    test_value_with_wo_padding()

    print("=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
