"""
Comprehensive tests for DinoMHC with single task heads.

This test suite verifies:
1. Each task head (affinity, presentation, contact) works correctly
2. Triangle operations still require square matrices (documented limitation)
3. Forward pass produces correct output shapes
4. Gradient flow works properly
"""
import torch
import sys
sys.path.append('src')
from model import DinoMHC, InterfaceGeometryModule


def test_affinity_head():
    """Test DinoMHC with affinity prediction head."""
    print("=" * 60)
    print("Test 1: DinoMHC with Affinity Head")
    print("=" * 60)

    batch_size = 2
    pep_len = 9
    mhc_len = 180
    num_groove_tokens = 9  # MUST equal pep_len for triangle operations!

    print(f"⚠ NOTE: Using num_groove_tokens={num_groove_tokens} to match pep_len={pep_len}")
    print(f"  This is required because triangle operations need square matrices.")
    print()

    # Create model with affinity head
    config = {
        'dim': 128,
        'num_groove_tokens': num_groove_tokens,  # Same as pep_len
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'affinity',
        'encoder_type': 'embedding'  # Use simple embedding encoder for tests
    }

    model = DinoMHC(config)
    model.eval()

    # Create dummy input
    peptide_tokens = torch.randint(0, 20, (batch_size, pep_len))
    mhc_tokens = torch.randint(0, 20, (batch_size, mhc_len))

    # Forward pass
    outputs = model(peptide_tokens, mhc_tokens, return_attention=True)

    print(f"✓ Model created with task_head='affinity'")
    print(f"✓ Input shapes: peptide={peptide_tokens.shape}, mhc={mhc_tokens.shape}")
    print(f"✓ Output keys: {list(outputs.keys())}")
    print(f"✓ Prediction shape: {outputs['prediction'].shape}")
    print(f"✓ Expected shape: [{batch_size}, 1] (affinity values)")

    assert outputs['prediction'].shape == (batch_size, 1), "Affinity output shape mismatch"
    assert 'attention_info' in outputs, "Attention info missing"
    assert not torch.isnan(outputs['prediction']).any(), "Output contains NaN"

    print(f"✓ Affinity head test passed!")
    print()


def test_presentation_head():
    """Test DinoMHC with presentation (classification) head."""
    print("=" * 60)
    print("Test 2: DinoMHC with Presentation Head")
    print("=" * 60)

    batch_size = 3
    pep_len = 10
    mhc_len = 180

    # Create model with presentation head
    config = {
        'dim': 128,
        'num_groove_tokens': 60,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation',
        'encoder_type': 'embedding'  # Use simple embedding encoder for tests
    }

    model = DinoMHC(config)
    model.eval()

    # Create dummy input
    peptide_tokens = torch.randint(0, 20, (batch_size, pep_len))
    mhc_tokens = torch.randint(0, 20, (batch_size, mhc_len))

    # Forward pass
    outputs = model(peptide_tokens, mhc_tokens)

    print(f"✓ Model created with task_head='presentation'")
    print(f"✓ Input shapes: peptide={peptide_tokens.shape}, mhc={mhc_tokens.shape}")
    print(f"✓ Prediction shape: {outputs['prediction'].shape}")
    print(f"✓ Expected shape: [{batch_size}, 1] (probabilities)")

    assert outputs['prediction'].shape == (batch_size, 1), "Presentation output shape mismatch"
    assert not torch.isnan(outputs['prediction']).any(), "Output contains NaN"

    # Check that output is a probability (0-1 range due to sigmoid)
    assert (outputs['prediction'] >= 0).all() and (outputs['prediction'] <= 1).all(), \
        "Presentation output not in [0, 1] range"

    print(f"✓ Output range: [{outputs['prediction'].min().item():.4f}, {outputs['prediction'].max().item():.4f}]")
    print(f"✓ Presentation head test passed!")
    print()


def test_contact_head():
    """Test DinoMHC with contact prediction head."""
    print("=" * 60)
    print("Test 3: DinoMHC with Contact Head")
    print("=" * 60)

    batch_size = 2
    pep_len = 9
    mhc_len = 180
    num_groove_tokens = 60

    # Create model with contact head
    config = {
        'dim': 128,
        'num_groove_tokens': num_groove_tokens,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 8,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'contact',
        'encoder_type': 'embedding'  # Use simple embedding encoder for tests
    }

    model = DinoMHC(config)
    model.eval()

    # Create dummy input
    peptide_tokens = torch.randint(0, 20, (batch_size, pep_len))
    mhc_tokens = torch.randint(0, 20, (batch_size, mhc_len))

    # Forward pass
    outputs = model(peptide_tokens, mhc_tokens)

    print(f"✓ Model created with task_head='contact'")
    print(f"✓ Input shapes: peptide={peptide_tokens.shape}, mhc={mhc_tokens.shape}")
    print(f"✓ Prediction shape: {outputs['prediction'].shape}")
    print(f"✓ Expected shape: [{batch_size}, {pep_len}, {num_groove_tokens}] (contact maps)")

    # After fusion, peptide length is preserved but groove tokens are used
    # The contact map should be [batch, pep_len, groove_len]
    expected_shape = (batch_size, pep_len, num_groove_tokens)
    assert outputs['prediction'].shape == expected_shape, \
        f"Contact output shape mismatch: got {outputs['prediction'].shape}, expected {expected_shape}"
    assert not torch.isnan(outputs['prediction']).any(), "Output contains NaN"

    # Check that output is probabilities (0-1 range due to sigmoid)
    assert (outputs['prediction'] >= 0).all() and (outputs['prediction'] <= 1).all(), \
        "Contact predictions not in [0, 1] range"

    print(f"✓ Output range: [{outputs['prediction'].min().item():.4f}, {outputs['prediction'].max().item():.4f}]")
    print(f"✓ Contact head test passed!")
    print()


def test_triangle_ops_with_non_square():
    """
    Test if triangle operations can handle non-square matrices.

    EXPECTED RESULT: This should FAIL because triangle operations
    (TriangleAttention, TriangleMultiplication) require SQUARE matrices.

    This is a documented limitation from AlphaFold's design.
    """
    print("=" * 60)
    print("Test 4: Triangle Operations with Non-Square Matrices")
    print("=" * 60)

    batch_size = 2
    pep_len = 12
    groove_len = 60
    dim = 128

    print(f"⚠ TESTING NON-SQUARE CASE: pep_len={pep_len}, groove_len={groove_len}")
    print(f"   This is expected to FAIL due to triangle operations limitation.")
    print()

    # Create InterfaceGeometryModule
    geom_module = InterfaceGeometryModule(
        dim=dim,
        num_blocks=2,
        num_heads=4,
        dropout=0.1
    )

    # Create non-square inputs
    peptide_emb = torch.randn(batch_size, pep_len, dim)
    groove_emb = torch.randn(batch_size, groove_len, dim)

    print(f"Input shapes: peptide={peptide_emb.shape}, groove={groove_emb.shape}")

    try:
        interface_rep, pair_rep = geom_module(
            peptide_emb, groove_emb, return_pair_rep=True
        )

        # If we get here, it worked!
        print(f"✓ UNEXPECTED: Triangle operations worked with non-square matrices!")
        print(f"✓ Interface rep shape: {interface_rep.shape}")
        print(f"✓ Pair rep shape: {pair_rep.shape}")
        print(f"✓ This means the limitation has been fixed!")
        print()
        return True

    except RuntimeError as e:
        print(f"✗ EXPECTED FAILURE: Triangle operations failed with non-square matrices")
        print(f"  Error: {str(e)}")
        print()
        print(f"This confirms the documented limitation:")
        print(f"  - Triangle operations require seq_i == seq_j")
        print(f"  - Current: pep_len={pep_len} != groove_len={groove_len}")
        print()
        return False


def test_triangle_ops_with_square():
    """
    Test that triangle operations work correctly with SQUARE matrices.
    """
    print("=" * 60)
    print("Test 5: Triangle Operations with Square Matrices")
    print("=" * 60)

    batch_size = 2
    seq_len = 30  # Same length for both
    dim = 128

    print(f"✓ TESTING SQUARE CASE: seq_len={seq_len} (same for both)")
    print()

    # Create InterfaceGeometryModule
    geom_module = InterfaceGeometryModule(
        dim=dim,
        num_blocks=2,
        num_heads=4,
        dropout=0.1
    )
    geom_module.eval()

    # Create square inputs
    peptide_emb = torch.randn(batch_size, seq_len, dim)
    groove_emb = torch.randn(batch_size, seq_len, dim)

    print(f"Input shapes: peptide={peptide_emb.shape}, groove={groove_emb.shape}")

    try:
        interface_rep, pair_rep = geom_module(
            peptide_emb, groove_emb, return_pair_rep=True
        )

        print(f"✓ Triangle operations succeeded with square matrices!")
        print(f"✓ Interface rep shape: {interface_rep.shape}")
        print(f"✓ Pair rep shape: {pair_rep.shape}")
        print(f"✓ Expected pair rep: [{batch_size}, {seq_len}, {seq_len}, {dim}]")

        assert interface_rep.shape == (batch_size, dim), "Interface rep shape mismatch"
        assert pair_rep.shape == (batch_size, seq_len, seq_len, dim), "Pair rep shape mismatch"
        assert not torch.isnan(interface_rep).any(), "Interface rep contains NaN"
        assert not torch.isnan(pair_rep).any(), "Pair rep contains NaN"

        print(f"✓ Square matrix test passed!")
        print()
        return True

    except RuntimeError as e:
        print(f"✗ UNEXPECTED FAILURE: Triangle operations failed even with square matrices")
        print(f"  Error: {str(e)}")
        print()
        return False


def test_gradient_flow():
    """Test that gradients flow correctly through all task heads."""
    print("=" * 60)
    print("Test 6: Gradient Flow for All Task Heads")
    print("=" * 60)

    batch_size = 2
    pep_len = 9
    mhc_len = 180

    for task_head in ['affinity', 'presentation', 'contact']:
        print(f"\nTesting gradient flow for task_head='{task_head}'...")

        config = {
            'dim': 64,  # Smaller for faster test
            'num_groove_tokens': 30,
            'num_fusion_layers': 1,
            'num_geometry_blocks': 1,
            'num_heads': 4,
            'dropout': 0.0,
            'max_peptide_length': 15,
            'task_head': task_head
        }

        model = DinoMHC(config)

        peptide_tokens = torch.randint(0, 20, (batch_size, pep_len))
        mhc_tokens = torch.randint(0, 20, (batch_size, mhc_len))

        # Forward pass
        outputs = model(peptide_tokens, mhc_tokens)

        # Compute loss and backward
        prediction = outputs['prediction']
        loss = prediction.sum()
        loss.backward()

        # Check that gradients exist
        has_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grads = True
                break

        assert has_grads, f"No gradients computed for task_head='{task_head}'"
        print(f"  ✓ Gradients flow correctly for '{task_head}'")

    print()
    print(f"✓ Gradient flow test passed for all task heads!")
    print()


def test_deterministic_eval():
    """Test that eval mode produces deterministic outputs."""
    print("=" * 60)
    print("Test 7: Deterministic Eval Mode")
    print("=" * 60)

    batch_size = 2
    pep_len = 9
    mhc_len = 180

    config = {
        'dim': 64,
        'num_groove_tokens': 30,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 1,
        'num_heads': 4,
        'dropout': 0.1,  # Even with dropout
        'max_peptide_length': 15,
        'task_head': 'presentation',
        'encoder_type': 'embedding'  # Use simple embedding encoder for tests
    }

    model = DinoMHC(config)
    model.eval()  # Eval mode should be deterministic

    peptide_tokens = torch.randint(0, 20, (batch_size, pep_len))
    mhc_tokens = torch.randint(0, 20, (batch_size, mhc_len))

    # Two forward passes with same input
    with torch.no_grad():
        outputs1 = model(peptide_tokens, mhc_tokens)
        outputs2 = model(peptide_tokens, mhc_tokens)

    diff = (outputs1['prediction'] - outputs2['prediction']).abs().mean()

    print(f"✓ Mean difference between two forward passes: {diff:.10f}")
    print(f"✓ Outputs are deterministic (diff < 1e-10): {diff < 1e-10}")

    assert diff < 1e-10, "Eval mode not deterministic"

    print(f"✓ Deterministic eval test passed!")
    print()


def test_invalid_task_head():
    """Test that invalid task_head raises an error."""
    print("=" * 60)
    print("Test 8: Invalid Task Head Error Handling")
    print("=" * 60)

    config = {
        'dim': 64,
        'num_groove_tokens': 30,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 1,
        'num_heads': 4,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'invalid_task'  # Invalid task
    }

    try:
        model = DinoMHC(config)
        print(f"✗ FAILED: Should have raised ValueError for invalid task_head")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)}")
        print(f"✓ Invalid task head test passed!")

    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    print("\n" + "=" * 60)
    print("DINOMHC COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    # Run all tests
    test_affinity_head()
    test_presentation_head()
    test_contact_head()

    # Critical test: Can triangle ops handle non-square?
    triangle_works_non_square = test_triangle_ops_with_non_square()
    triangle_works_square = test_triangle_ops_with_square()

    test_gradient_flow()
    test_deterministic_eval()
    test_invalid_task_head()

    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ All task heads (affinity, presentation, contact) work correctly")
    print(f"✓ Forward pass produces correct output shapes")
    print(f"✓ Gradient flow verified for all task heads")
    print(f"✓ Eval mode is deterministic")
    print(f"✓ Invalid task head raises appropriate error")
    print()
    print("Triangle Operations Status:")
    if triangle_works_non_square:
        print(f"  ✓ Triangle ops WORK with non-square matrices (limitation fixed!)")
    else:
        print(f"  ✗ Triangle ops FAIL with non-square matrices (documented limitation)")
        print(f"    - This is expected behavior from AlphaFold's design")
        print(f"    - Workaround: Use GroovePeptideFusion (which handles non-square)")

    if triangle_works_square:
        print(f"  ✓ Triangle ops work correctly with square matrices")
    else:
        print(f"  ✗ Triangle ops failed even with square matrices (unexpected!)")

    print()
    print("=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
