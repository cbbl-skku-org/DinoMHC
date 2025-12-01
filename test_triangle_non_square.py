"""
Test the updated InterfaceGeometryModule with padding for non-square matrices.
"""
import torch
import sys
sys.path.append('src')
from model import (
    TriangleMultiplication, 
    TriangleAttention,
    InterfaceGeometryBlock,
    InterfaceGeometryModule
)

print("=" * 70)
print("TESTING TRIANGLE OPERATIONS WITH PADDING AND MASKING")
print("=" * 70)
print()

# Test 1: TriangleMultiplication with mask (square padded)
print("Test 1: TriangleMultiplication with Mask")
print("-" * 70)
batch_size = 2
max_len = 60  # padded square size
pep_len = 12
groove_len = 60
dim = 64

module = TriangleMultiplication(dim, mode='outgoing', dropout=0.1)
module.eval()

# Create padded square matrix
pair_rep = torch.randn(batch_size, max_len, max_len, dim)

# Create mask: True for valid [0:pep_len, 0:groove_len], False elsewhere
pair_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool)
pair_mask[:, :pep_len, :groove_len] = True

print(f"Padded input shape: {pair_rep.shape}")
print(f"Mask shape: {pair_mask.shape}")
print(f"Valid region: [{pep_len}, {groove_len}]")

try:
    output = module(pair_rep, pair_mask)
    print(f"✓ Output shape: {output.shape}")
    
    # Check that padded positions are properly handled
    valid_output = output[:, :pep_len, :groove_len, :]
    print(f"✓ Valid output region has non-zero values: {valid_output.abs().mean():.4f}")
    print("✓ SUCCESS: TriangleMultiplication works with mask!\n")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}\n")

# Test 2: TriangleAttention with mask
print("Test 2: TriangleAttention with Mask")
print("-" * 70)
attn_module = TriangleAttention(dim, num_heads=4, mode='starting', dropout=0.1)
attn_module.eval()

try:
    output_attn = attn_module(pair_rep, pair_mask)
    print(f"✓ Output shape: {output_attn.shape}")
    print("✓ SUCCESS: TriangleAttention works with mask!\n")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}\n")

# Test 3: InterfaceGeometryBlock with mask
print("Test 3: InterfaceGeometryBlock with Mask")
print("-" * 70)
block = InterfaceGeometryBlock(dim, num_heads=4, dropout=0.1)
block.eval()

try:
    output_block = block(pair_rep, pair_mask)
    print(f"✓ Output shape: {output_block.shape}")
    print("✓ SUCCESS: InterfaceGeometryBlock works with mask!\n")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}\n")

# Test 4: Full InterfaceGeometryModule with non-square inputs
print("Test 4: InterfaceGeometryModule with Non-Square Inputs")
print("-" * 70)
pep_len = 12
groove_len = 60
dim = 128

geom_module = InterfaceGeometryModule(
    dim=dim, 
    num_blocks=2, 
    num_heads=4, 
    dropout=0.1
)
geom_module.eval()

peptide_emb = torch.randn(batch_size, pep_len, dim)
groove_emb = torch.randn(batch_size, groove_len, dim)

print(f"Peptide input shape: {peptide_emb.shape}")
print(f"Groove input shape: {groove_emb.shape}")

try:
    interface_rep, pair_rep_out = geom_module(
        peptide_emb, groove_emb, return_pair_rep=True
    )
    print(f"✓ Interface representation shape: {interface_rep.shape}")
    print(f"✓ Pair representation shape: {pair_rep_out.shape}")
    print(f"✓ Expected pair shape: [{batch_size}, {pep_len}, {groove_len}, {dim}]")
    
    # Verify shapes
    assert interface_rep.shape == (batch_size, dim), "Wrong interface shape"
    assert pair_rep_out.shape == (batch_size, pep_len, groove_len, dim), "Wrong pair shape"
    
    # Check for NaN
    assert not torch.isnan(interface_rep).any(), "Interface rep contains NaN"
    assert not torch.isnan(pair_rep_out).any(), "Pair rep contains NaN"
    
    print("✓ SUCCESS: InterfaceGeometryModule works with non-square inputs!\n")
except Exception as e:
    import traceback
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print()

# Test 5: Gradient flow with non-square inputs
print("Test 5: Gradient Flow with Non-Square Inputs")
print("-" * 70)
peptide_emb = torch.randn(batch_size, pep_len, dim, requires_grad=True)
groove_emb = torch.randn(batch_size, groove_len, dim, requires_grad=True)

try:
    interface_rep, _ = geom_module(peptide_emb, groove_emb)
    loss = interface_rep.sum()
    loss.backward()
    
    assert peptide_emb.grad is not None, "No gradient for peptide"
    assert groove_emb.grad is not None, "No gradient for groove"
    
    print(f"✓ Peptide gradient norm: {peptide_emb.grad.norm():.4f}")
    print(f"✓ Groove gradient norm: {groove_emb.grad.norm():.4f}")
    print("✓ SUCCESS: Gradients flow correctly!\n")
except Exception as e:
    import traceback
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print()

# Test 6: Various aspect ratios
print("Test 6: Various Aspect Ratios")
print("-" * 70)
test_cases = [
    (9, 60, "Typical 9-mer peptide"),
    (12, 60, "12-mer peptide"),
    (15, 60, "15-mer peptide"),
    (30, 30, "Square (equal lengths)"),
    (60, 12, "Inverted (pep > groove)"),
]

for pep_l, groove_l, desc in test_cases:
    peptide = torch.randn(batch_size, pep_l, dim)
    groove = torch.randn(batch_size, groove_l, dim)
    
    try:
        interface, pair = geom_module(peptide, groove, return_pair_rep=True)
        status = "✓"
        msg = f"interface: {interface.shape}, pair: {pair.shape}"
    except Exception as e:
        status = "✗"
        msg = str(e)[:50]
    
    print(f"  {status} [{pep_l:3d} x {groove_l:3d}] {desc}: {msg}")

print()
print("=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)
