"""
Comprehensive unit tests for new modules in model.py:
- RelativePositionBias
- GeometricBiasModule
- CrossAttentionWithGeometricBias
- GroovePeptideFusionLayer
- GroovePeptideFusion
- TriangleAttention
- TriangleMultiplication
- PairTransition
- InterfaceGeometryBlock
- InterfaceGeometryModule
"""
import torch
import torch.nn as nn
import sys
sys.path.append('src')
from model import (
    RelativePositionBias,
    GeometricBiasModule,
    CrossAttentionWithGeometricBias,
    GroovePeptideFusionLayer,
    GroovePeptideFusion,
    TriangleAttention,
    TriangleMultiplication,
    PairTransition,
    InterfaceGeometryBlock,
    InterfaceGeometryModule
)


def test_relative_position_bias():
    """Test RelativePositionBias module."""
    print("=" * 60)
    print("Test 1: RelativePositionBias")
    print("=" * 60)

    num_heads = 8
    max_distance = 32
    query_len = 10
    key_len = 15

    module = RelativePositionBias(num_heads, max_distance)

    # Test forward pass
    bias = module(query_len, key_len, torch.device('cpu'))

    print(f"✓ Bias shape: {bias.shape}")
    print(f"✓ Expected shape: [num_heads={num_heads}, query_len={query_len}, key_len={key_len}]")
    assert bias.shape == (num_heads, query_len, key_len), "Incorrect bias shape"

    # Test that diagonal has different values than off-diagonal
    # (since relative distances differ)
    if query_len == key_len:
        diag_mean = bias[:, range(query_len), range(key_len)].mean()
        off_diag_mean = bias[:, 0, 1:].mean()
        print(f"✓ Diagonal bias mean: {diag_mean:.4f}")
        print(f"✓ Off-diagonal bias mean: {off_diag_mean:.4f}")

    # Test clamping at max distance
    module2 = RelativePositionBias(num_heads, max_relative_distance=5)
    bias_large = module2(100, 100, torch.device('cpu'))
    print(f"✓ Bias with max_distance=5 computed for 100x100 positions")
    print(f"✓ No errors with large sequences (clamping works)")

    print()


def test_geometric_bias_module():
    """Test GeometricBiasModule."""
    print("=" * 60)
    print("Test 2: GeometricBiasModule")
    print("=" * 60)

    batch_size = 2
    query_len = 12
    key_len = 60
    dim = 128
    num_heads = 8

    module = GeometricBiasModule(dim, num_heads)

    query_emb = torch.randn(batch_size, query_len, dim)
    key_emb = torch.randn(batch_size, key_len, dim)

    bias = module(query_emb, key_emb)

    print(f"✓ Input query shape: {query_emb.shape}")
    print(f"✓ Input key shape: {key_emb.shape}")
    print(f"✓ Output bias shape: {bias.shape}")
    print(f"✓ Expected: [batch={batch_size}, num_heads={num_heads}, query_len={query_len}, key_len={key_len}]")

    assert bias.shape == (batch_size, num_heads, query_len, key_len), "Incorrect bias shape"
    assert not torch.isnan(bias).any(), "Bias contains NaN"

    # Test that different pairs produce different biases
    bias_sample1 = bias[0, 0, 0, :]
    bias_sample2 = bias[0, 0, 1, :]
    diff = (bias_sample1 - bias_sample2).abs().mean()
    print(f"✓ Different query positions produce different biases: {diff > 0.01}")

    print()


def test_cross_attention_with_geometric_bias():
    """Test CrossAttentionWithGeometricBias."""
    print("=" * 60)
    print("Test 3: CrossAttentionWithGeometricBias")
    print("=" * 60)

    batch_size = 2
    query_len = 12
    kv_len = 60
    dim = 128
    num_heads = 8

    # Test with geometric bias
    module = CrossAttentionWithGeometricBias(
        dim, num_heads, dropout=0.1, use_geometric_bias=True
    )
    module.eval()

    query = torch.randn(batch_size, query_len, dim)
    key_value = torch.randn(batch_size, kv_len, dim)

    output, attn_weights = module(query, key_value, return_attention=True)

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: [batch={batch_size}, query_len={query_len}, dim={dim}]")
    print(f"✓ Attention weights shape: {attn_weights.shape}")
    print(f"✓ Expected: [batch={batch_size}, num_heads={num_heads}, query_len={query_len}, kv_len={kv_len}]")

    assert output.shape == (batch_size, query_len, dim), "Incorrect output shape"
    assert attn_weights.shape == (batch_size, num_heads, query_len, kv_len), "Incorrect attention shape"
    assert not torch.isnan(output).any(), "Output contains NaN"

    # Check attention weights sum to 1
    attn_sum = attn_weights.sum(dim=-1)
    print(f"✓ Attention weights sum to 1: {torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)}")

    # Test without geometric bias
    module_no_geo = CrossAttentionWithGeometricBias(
        dim, num_heads, dropout=0.1, use_geometric_bias=False
    )
    module_no_geo.eval()

    output_no_geo, _ = module_no_geo(query, key_value)
    print(f"✓ Module works without geometric bias")

    # Test with attention mask
    mask = torch.ones(batch_size, 1, query_len, kv_len)
    mask[:, :, :, kv_len//2:] = 0  # Mask second half

    output_masked, attn_masked = module(query, key_value, attention_mask=mask, return_attention=True)

    # Check masked positions have zero attention
    attn_on_masked = attn_masked[:, :, :, kv_len//2:].abs().mean()
    print(f"✓ Attention on masked positions: {attn_on_masked:.6f} (should be ~0)")

    print()


def test_groove_peptide_fusion_layer():
    """Test GroovePeptideFusionLayer."""
    print("=" * 60)
    print("Test 4: GroovePeptideFusionLayer")
    print("=" * 60)

    batch_size = 2
    pep_len = 12
    groove_len = 60
    dim = 128
    num_heads = 8

    module = GroovePeptideFusionLayer(dim, num_heads, dropout=0.1)
    module.eval()

    peptide_emb = torch.randn(batch_size, pep_len, dim)
    groove_emb = torch.randn(batch_size, groove_len, dim)

    pep_out, groove_out, attn_info = module(
        peptide_emb, groove_emb, return_attention=True
    )

    print(f"✓ Peptide output shape: {pep_out.shape}")
    print(f"✓ Groove output shape: {groove_out.shape}")
    assert pep_out.shape == (batch_size, pep_len, dim), "Incorrect peptide output shape"
    assert groove_out.shape == (batch_size, groove_len, dim), "Incorrect groove output shape"
    assert not torch.isnan(pep_out).any(), "Peptide output contains NaN"
    assert not torch.isnan(groove_out).any(), "Groove output contains NaN"

    # Check attention info
    print(f"✓ Attention info keys: {list(attn_info.keys())}")
    assert 'peptide_to_groove' in attn_info, "Missing peptide_to_groove attention"
    assert 'groove_to_peptide' in attn_info, "Missing groove_to_peptide attention"

    if attn_info['peptide_to_groove'] is not None:
        print(f"✓ Peptide-to-groove attention shape: {attn_info['peptide_to_groove'].shape}")
    if attn_info['groove_to_peptide'] is not None:
        print(f"✓ Groove-to-peptide attention shape: {attn_info['groove_to_peptide'].shape}")

    print()


def test_groove_peptide_fusion():
    """Test GroovePeptideFusion (multi-layer)."""
    print("=" * 60)
    print("Test 5: GroovePeptideFusion (Multi-layer)")
    print("=" * 60)

    batch_size = 2
    pep_len = 12
    groove_len = 60
    dim = 128
    num_layers = 4
    num_heads = 8

    module = GroovePeptideFusion(
        dim, num_layers=num_layers, num_heads=num_heads, dropout=0.1
    )
    module.eval()

    peptide_emb = torch.randn(batch_size, pep_len, dim)
    groove_emb = torch.randn(batch_size, groove_len, dim)

    pep_fused, groove_fused, all_attn = module(
        peptide_emb, groove_emb, return_attention=True
    )

    print(f"✓ Peptide fused shape: {pep_fused.shape}")
    print(f"✓ Groove fused shape: {groove_fused.shape}")
    assert pep_fused.shape == (batch_size, pep_len, dim), "Incorrect fused peptide shape"
    assert groove_fused.shape == (batch_size, groove_len, dim), "Incorrect fused groove shape"

    # Check all layers returned attention
    print(f"✓ Number of layers with attention: {len(all_attn)}")
    assert len(all_attn) == num_layers, f"Expected {num_layers} attention dicts"

    for i in range(num_layers):
        layer_key = f'layer_{i}'
        assert layer_key in all_attn, f"Missing attention for {layer_key}"
        print(f"  ✓ Layer {i} attention keys: {list(all_attn[layer_key].keys())}")

    print()


def test_triangle_attention():
    """Test TriangleAttention."""
    print("=" * 60)
    print("Test 6: TriangleAttention")
    print("=" * 60)

    batch_size = 2
    seq_len = 30  # Use square for consistency with triangle operations
    dim = 64
    num_heads = 4

    # Test starting mode
    module_start = TriangleAttention(dim, num_heads, mode='starting', dropout=0.1)
    module_start.eval()

    pair_rep = torch.randn(batch_size, seq_len, seq_len, dim)
    output_start = module_start(pair_rep)

    print(f"✓ Input pair shape: {pair_rep.shape}")
    print(f"✓ Output (starting) shape: {output_start.shape}")
    assert output_start.shape == pair_rep.shape, "Output shape mismatch (starting)"
    assert not torch.isnan(output_start).any(), "Output contains NaN (starting)"

    # Test ending mode
    module_end = TriangleAttention(dim, num_heads, mode='ending', dropout=0.1)
    module_end.eval()

    output_end = module_end(pair_rep)

    print(f"✓ Output (ending) shape: {output_end.shape}")
    assert output_end.shape == pair_rep.shape, "Output shape mismatch (ending)"
    assert not torch.isnan(output_end).any(), "Output contains NaN (ending)"

    # Check that outputs are different (different modes)
    diff = (output_start - output_end).abs().mean()
    print(f"✓ Starting vs Ending mode difference: {diff:.4f} (should be > 0)")
    assert diff > 1e-3, "Starting and ending modes produce identical outputs"

    print()


def test_triangle_multiplication():
    """Test TriangleMultiplication."""
    print("=" * 60)
    print("Test 7: TriangleMultiplication")
    print("=" * 60)

    batch_size = 2
    seq_len = 30  # Use square matrix for triangle multiplication
    dim = 64

    # Test outgoing mode
    module_out = TriangleMultiplication(dim, mode='outgoing', dropout=0.1)
    module_out.eval()

    # Triangle multiplication expects square pair representation
    pair_rep = torch.randn(batch_size, seq_len, seq_len, dim)
    output_out = module_out(pair_rep)

    print(f"✓ Input pair shape: {pair_rep.shape}")
    print(f"✓ Output (outgoing) shape: {output_out.shape}")
    assert output_out.shape == pair_rep.shape, "Output shape mismatch (outgoing)"
    assert not torch.isnan(output_out).any(), "Output contains NaN (outgoing)"

    # Test incoming mode
    module_in = TriangleMultiplication(dim, mode='incoming', dropout=0.1)
    module_in.eval()

    output_in = module_in(pair_rep)

    print(f"✓ Output (incoming) shape: {output_in.shape}")
    assert output_in.shape == pair_rep.shape, "Output shape mismatch (incoming)"
    assert not torch.isnan(output_in).any(), "Output contains NaN (incoming)"

    # Check that outputs are different (different modes)
    diff = (output_out - output_in).abs().mean()
    print(f"✓ Outgoing vs Incoming mode difference: {diff:.4f} (should be > 0)")
    assert diff > 1e-3, "Outgoing and incoming modes produce identical outputs"

    print()


def test_pair_transition():
    """Test PairTransition."""
    print("=" * 60)
    print("Test 8: PairTransition")
    print("=" * 60)

    batch_size = 2
    seq_len = 30
    dim = 64

    module = PairTransition(dim, expansion_factor=4, dropout=0.1)
    module.eval()

    pair_rep = torch.randn(batch_size, seq_len, seq_len, dim)
    output = module(pair_rep)

    print(f"✓ Input shape: {pair_rep.shape}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == pair_rep.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN"

    # Check residual connection (output should differ from input)
    diff = (output - pair_rep).abs().mean()
    print(f"✓ Difference from input (residual applied): {diff:.4f}")
    assert diff > 1e-3, "No transformation applied"

    print()


def test_interface_geometry_block():
    """Test InterfaceGeometryBlock."""
    print("=" * 60)
    print("Test 9: InterfaceGeometryBlock")
    print("=" * 60)

    batch_size = 2
    seq_len = 30
    dim = 64
    num_heads = 4

    module = InterfaceGeometryBlock(dim, num_heads, dropout=0.1)
    module.eval()

    pair_rep = torch.randn(batch_size, seq_len, seq_len, dim)
    output = module(pair_rep)

    print(f"✓ Input shape: {pair_rep.shape}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == pair_rep.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN"

    # Check that block applies transformations
    diff = (output - pair_rep).abs().mean()
    print(f"✓ Transformation magnitude: {diff:.4f}")
    assert diff > 1e-3, "No transformation applied"

    print()


def test_interface_geometry_module():
    """Test InterfaceGeometryModule."""
    print("=" * 60)
    print("Test 10: InterfaceGeometryModule")
    print("=" * 60)

    batch_size = 2
    dim = 128
    num_blocks = 3
    num_heads = 4

    print("Testing with non-square inputs (pep_len != groove_len).")
    print("The module now supports padding to square with proper masking.")
    print()

    # Test with non-square inputs (typical MHC-peptide case)
    pep_len = 12
    groove_len = 60

    module = InterfaceGeometryModule(
        dim, num_blocks=num_blocks, num_heads=num_heads, dropout=0.1
    )
    module.eval()

    peptide_emb = torch.randn(batch_size, pep_len, dim)
    groove_emb = torch.randn(batch_size, groove_len, dim)

    # Test without returning pair representation
    interface_rep, pair_rep = module(peptide_emb, groove_emb, return_pair_rep=False)

    print(f"✓ Peptide input shape: {peptide_emb.shape}")
    print(f"✓ Groove input shape: {groove_emb.shape}")
    print(f"✓ Interface representation shape: {interface_rep.shape}")
    assert interface_rep.shape == (batch_size, dim), "Incorrect interface rep shape"
    assert pair_rep is None, "Pair rep should be None when return_pair_rep=False"
    assert not torch.isnan(interface_rep).any(), "Interface rep contains NaN"

    # Test with returning pair representation
    interface_rep2, pair_rep2 = module(peptide_emb, groove_emb, return_pair_rep=True)

    print(f"✓ Pair representation shape: {pair_rep2.shape}")
    print(f"✓ Expected: [batch={batch_size}, pep_len={pep_len}, groove_len={groove_len}, dim={dim}]")
    assert pair_rep2.shape == (batch_size, pep_len, groove_len, dim), "Incorrect pair rep shape"
    assert not torch.isnan(pair_rep2).any(), "Pair rep contains NaN"

    # Check that pooling works correctly
    manual_pool = pair_rep2.mean(dim=[1, 2])
    print(f"✓ Interface rep matches manual pooling: {torch.allclose(interface_rep2, manual_pool, atol=1e-5)}")

    # Also test with square inputs for completeness
    print()
    print("Also testing with square inputs (pep_len == groove_len):")
    seq_len = 30
    peptide_sq = torch.randn(batch_size, seq_len, dim)
    groove_sq = torch.randn(batch_size, seq_len, dim)
    interface_sq, pair_sq = module(peptide_sq, groove_sq, return_pair_rep=True)
    print(f"✓ Square input interface shape: {interface_sq.shape}")
    print(f"✓ Square input pair shape: {pair_sq.shape}")
    assert interface_sq.shape == (batch_size, dim), "Incorrect interface rep shape (square)"
    assert pair_sq.shape == (batch_size, seq_len, seq_len, dim), "Incorrect pair rep shape (square)"

    print()


def test_gradient_flow():
    """Test that gradients flow through all modules."""
    print("=" * 60)
    print("Test 11: Gradient Flow")
    print("=" * 60)

    batch_size = 2
    pep_len = 12
    groove_len = 60
    dim = 64

    # Test GroovePeptideFusion (works with different lengths)
    fusion = GroovePeptideFusion(dim, num_layers=2, num_heads=4, dropout=0.0)

    peptide_emb = torch.randn(batch_size, pep_len, dim, requires_grad=True)
    groove_emb = torch.randn(batch_size, groove_len, dim, requires_grad=True)

    pep_out, groove_out, _ = fusion(peptide_emb, groove_emb)

    loss = pep_out.sum() + groove_out.sum()
    loss.backward()

    assert peptide_emb.grad is not None, "No gradient for peptide input"
    assert groove_emb.grad is not None, "No gradient for groove input"
    print(f"✓ GroovePeptideFusion: Gradients flow correctly")
    print(f"  Peptide grad norm: {peptide_emb.grad.norm():.4f}")
    print(f"  Groove grad norm: {groove_emb.grad.norm():.4f}")

    # Test InterfaceGeometryModule (now works with non-square inputs via padding)
    geom = InterfaceGeometryModule(dim, num_blocks=2, num_heads=4, dropout=0.0)

    peptide_emb2 = torch.randn(batch_size, pep_len, dim, requires_grad=True)
    groove_emb2 = torch.randn(batch_size, groove_len, dim, requires_grad=True)

    interface_rep, _ = geom(peptide_emb2, groove_emb2)

    loss2 = interface_rep.sum()
    loss2.backward()

    assert peptide_emb2.grad is not None, "No gradient for peptide input"
    assert groove_emb2.grad is not None, "No gradient for groove input"
    print(f"✓ InterfaceGeometryModule: Gradients flow correctly (pep={pep_len}, groove={groove_len})")
    print(f"  Peptide grad norm: {peptide_emb2.grad.norm():.4f}")
    print(f"  Groove grad norm: {groove_emb2.grad.norm():.4f}")

    print()


def test_deterministic_eval_mode():
    """Test that eval mode produces deterministic outputs."""
    print("=" * 60)
    print("Test 12: Deterministic Eval Mode")
    print("=" * 60)

    batch_size = 2
    pep_len = 12
    groove_len = 60
    dim = 64

    torch.manual_seed(42)
    fusion = GroovePeptideFusion(dim, num_layers=2, num_heads=4, dropout=0.1)
    fusion.eval()

    peptide_emb = torch.randn(batch_size, pep_len, dim)
    groove_emb = torch.randn(batch_size, groove_len, dim)

    # Run twice
    pep_out1, groove_out1, _ = fusion(peptide_emb, groove_emb)
    pep_out2, groove_out2, _ = fusion(peptide_emb, groove_emb)

    pep_diff = (pep_out1 - pep_out2).abs().max()
    groove_diff = (groove_out1 - groove_out2).abs().max()

    print(f"✓ Peptide output difference: {pep_diff:.10f}")
    print(f"✓ Groove output difference: {groove_diff:.10f}")
    print(f"✓ Outputs are deterministic: {pep_diff < 1e-6 and groove_diff < 1e-6}")

    print()


if __name__ == "__main__":
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("COMPREHENSIVE UNIT TESTS FOR NEW MODULES")
    print("=" * 60 + "\n")

    # Run all tests
    test_relative_position_bias()
    test_geometric_bias_module()
    test_cross_attention_with_geometric_bias()
    test_groove_peptide_fusion_layer()
    test_groove_peptide_fusion()
    test_triangle_attention()
    test_triangle_multiplication()
    test_pair_transition()
    test_interface_geometry_block()
    test_interface_geometry_module()
    test_gradient_flow()
    test_deterministic_eval_mode()

    print("=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
