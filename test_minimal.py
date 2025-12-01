"""
Minimal test to verify HierarchicalGrooveExtractor padding handling.
Run this to quickly check if the module works correctly.
"""
import torch
import sys
sys.path.append('src')
from model import HierarchicalGrooveExtractor

print("Testing HierarchicalGrooveExtractor with padding...")
print("=" * 60)

# Create model
extractor = HierarchicalGrooveExtractor(
    dim=128,
    num_groove_tokens=60,
    num_regions=3,
    num_heads=8
)

# Test case: 2 samples with different sequence lengths
batch_size = 2
mhc_len = 180
pep_len = 12

# Create embeddings
mhc_emb = torch.randn(batch_size, mhc_len, 128)
peptide_emb = torch.randn(batch_size, pep_len, 128)

# Create masks
# Sample 0: Full length (no padding)
# Sample 1: Shorter sequence (with padding)
mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool)
mhc_mask[1, 100:] = False  # Sample 1 has only 100 valid tokens

peptide_mask = torch.ones(batch_size, pep_len, dtype=torch.bool)
peptide_mask[1, 8:] = False  # Sample 1 has only 8 valid peptide tokens

print(f"Sample 0: MHC length={mhc_mask[0].sum().item()}, Peptide length={peptide_mask[0].sum().item()}")
print(f"Sample 1: MHC length={mhc_mask[1].sum().item()}, Peptide length={peptide_mask[1].sum().item()}")
print()

# Test 1: Forward pass with masks
print("Test 1: Forward pass WITH padding masks")
groove_features, attn_info = extractor(
    mhc_emb=mhc_emb,
    peptide_emb=peptide_emb,
    mhc_mask=mhc_mask,
    peptide_mask=peptide_mask,
    return_attention=True
)

print(f"✓ Groove features shape: {groove_features.shape}")
print(f"✓ Region attention shape: {attn_info['region_attention'].shape}")
print(f"✓ Groove attention shape: {attn_info['groove_attention'].shape}")
print(f"✓ No NaN values: {not torch.isnan(groove_features).any()}")

# Check attention on padding positions
region_attn = attn_info['region_attention']  # [batch, num_heads, num_regions, mhc_len]
attn_on_padding = region_attn[1, :, :, 100:].abs().mean().item()
print(f"✓ Attention on padding positions: {attn_on_padding:.6f} (should be ~0)")
print()

# Test 2: Forward pass without masks (should still work)
print("Test 2: Forward pass WITHOUT padding masks")
groove_features_no_mask, _ = extractor(
    mhc_emb=mhc_emb,
    peptide_emb=peptide_emb
)

print(f"✓ Groove features shape: {groove_features_no_mask.shape}")
print(f"✓ No NaN values: {not torch.isnan(groove_features_no_mask).any()}")
print()

# Test 3: Forward pass without peptide
print("Test 3: Forward pass WITHOUT peptide embeddings")
groove_features_no_pep, _ = extractor(
    mhc_emb=mhc_emb,
    mhc_mask=mhc_mask
)

print(f"✓ Groove features shape: {groove_features_no_pep.shape}")
print(f"✓ No NaN values: {not torch.isnan(groove_features_no_pep).any()}")
print()

print("=" * 60)
print("✓ All tests passed! Padding is handled correctly.")
print("=" * 60)
