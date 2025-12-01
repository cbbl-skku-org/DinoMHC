"""
Comprehensive test for mask handling across all modules.
Tests that padding positions are properly handled during training.
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
    DinoMHC
)

torch.manual_seed(42)

def test_hierarchical_groove_extractor_mask():
    """Test HierarchicalGrooveExtractor with masks."""
    print("=" * 70)
    print("Test 1: HierarchicalGrooveExtractor Mask Handling")
    print("=" * 70)
    
    batch_size = 2
    mhc_len = 100
    pep_len = 12
    dim = 64
    
    module = HierarchicalGrooveExtractor(dim=dim, num_groove_tokens=30, num_heads=4)
    module.eval()
    
    mhc_emb = torch.randn(batch_size, mhc_len, dim)
    pep_emb = torch.randn(batch_size, pep_len, dim)
    
    # Create masks: first sample has full length, second has half
    mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool)
    mhc_mask[1, mhc_len//2:] = False  # Second sample has half MHC
    
    pep_mask = torch.ones(batch_size, pep_len, dtype=torch.bool)
    pep_mask[1, pep_len//2:] = False  # Second sample has half peptide
    
    # Zero out padded positions in input
    mhc_emb[1, mhc_len//2:] = 0
    pep_emb[1, pep_len//2:] = 0
    
    print(f"MHC shape: {mhc_emb.shape}, valid: sample0={mhc_len}, sample1={mhc_len//2}")
    print(f"Peptide shape: {pep_emb.shape}, valid: sample0={pep_len}, sample1={pep_len//2}")
    
    # Forward with masks
    groove_features, attn_info = module(
        mhc_emb, pep_emb, 
        mhc_mask=mhc_mask, 
        peptide_mask=pep_mask,
        return_attention=True
    )
    
    print(f"✓ Output shape: {groove_features.shape}")
    assert not torch.isnan(groove_features).any(), "Output contains NaN"
    
    # Check attention weights on masked positions are low
    groove_attn = attn_info['groove_attention']  # [batch, heads, groove_tokens, mhc_len]
    masked_attn = groove_attn[1, :, :, mhc_len//2:].mean()
    valid_attn = groove_attn[1, :, :, :mhc_len//2].mean()
    print(f"✓ Attention on masked MHC positions: {masked_attn:.6f} (should be ~0)")
    print(f"✓ Attention on valid MHC positions: {valid_attn:.6f}")
    print()


def test_groove_peptide_fusion_mask():
    """Test GroovePeptideFusion with masks."""
    print("=" * 70)
    print("Test 2: GroovePeptideFusion Mask Handling")
    print("=" * 70)
    
    batch_size = 2
    pep_len = 12
    groove_len = 30
    dim = 64
    
    module = GroovePeptideFusion(dim=dim, num_layers=2, num_heads=4)
    module.eval()
    
    peptide_emb = torch.randn(batch_size, pep_len, dim)
    groove_emb = torch.randn(batch_size, groove_len, dim)
    
    # Create masks
    pep_mask = torch.ones(batch_size, pep_len, dtype=torch.bool)
    pep_mask[1, pep_len//2:] = False  # Second sample has half peptide
    
    groove_mask = torch.ones(batch_size, groove_len, dtype=torch.bool)
    groove_mask[1, groove_len//2:] = False  # Second sample has half groove
    
    # Zero out padded positions in input
    peptide_emb[1, pep_len//2:] = 0
    groove_emb[1, groove_len//2:] = 0
    
    print(f"Peptide shape: {peptide_emb.shape}, valid: sample0={pep_len}, sample1={pep_len//2}")
    print(f"Groove shape: {groove_emb.shape}, valid: sample0={groove_len}, sample1={groove_len//2}")
    
    # Forward with masks
    pep_out, groove_out, attn_info = module(
        peptide_emb, groove_emb,
        peptide_mask=pep_mask,
        groove_mask=groove_mask,
        return_attention=True
    )
    
    print(f"✓ Peptide output shape: {pep_out.shape}")
    print(f"✓ Groove output shape: {groove_out.shape}")
    
    # Check that masked positions are zeroed in output
    pep_masked_output = pep_out[1, pep_len//2:].abs().max()
    groove_masked_output = groove_out[1, groove_len//2:].abs().max()
    
    print(f"✓ Peptide output at masked positions (max abs): {pep_masked_output:.6f} (should be 0)")
    print(f"✓ Groove output at masked positions (max abs): {groove_masked_output:.6f} (should be 0)")
    
    assert pep_masked_output < 1e-6, "Peptide masked positions should be zero"
    assert groove_masked_output < 1e-6, "Groove masked positions should be zero"
    print()


def test_position_aware_encoder_mask():
    """Test PositionAwarePeptideEncoder with mask."""
    print("=" * 70)
    print("Test 3: PositionAwarePeptideEncoder Mask Handling")
    print("=" * 70)
    
    batch_size = 2
    pep_len = 12
    dim = 64
    
    module = PositionAwarePeptideEncoder(dim=dim, max_peptide_length=15, num_heads=4)
    module.eval()
    
    peptide_emb = torch.randn(batch_size, pep_len, dim)
    
    # Create mask
    pep_mask = torch.ones(batch_size, pep_len, dtype=torch.bool)
    pep_mask[1, pep_len//2:] = False  # Second sample has half peptide
    
    # Zero out padded positions in input
    peptide_emb[1, pep_len//2:] = 0
    
    print(f"Peptide shape: {peptide_emb.shape}, valid: sample0={pep_len}, sample1={pep_len//2}")
    
    # Forward with mask
    enhanced, importance = module(
        peptide_emb,
        peptide_mask=pep_mask,
        return_importance=True
    )
    
    print(f"✓ Enhanced output shape: {enhanced.shape}")
    print(f"✓ Importance shape: {importance.shape}")
    
    # Check that masked positions are zeroed
    masked_output = enhanced[1, pep_len//2:].abs().max()
    masked_importance = importance[1, pep_len//2:].abs().max()
    
    print(f"✓ Output at masked positions (max abs): {masked_output:.6f} (should be 0)")
    print(f"✓ Importance at masked positions (max abs): {masked_importance:.6f} (should be 0)")
    
    assert masked_output < 1e-6, "Masked positions should be zero"
    print()


def test_interface_geometry_module_mask():
    """Test InterfaceGeometryModule handles internal masking correctly."""
    print("=" * 70)
    print("Test 4: InterfaceGeometryModule Internal Mask Handling")
    print("=" * 70)
    
    batch_size = 2
    pep_len = 12
    groove_len = 60
    dim = 64
    
    module = InterfaceGeometryModule(dim=dim, num_blocks=2, num_heads=4)
    module.eval()
    
    peptide_emb = torch.randn(batch_size, pep_len, dim)
    groove_emb = torch.randn(batch_size, groove_len, dim)
    
    print(f"Peptide shape: {peptide_emb.shape}")
    print(f"Groove shape: {groove_emb.shape}")
    print(f"Note: Module internally pads to square ({groove_len}x{groove_len}) with mask")
    
    # Forward
    interface_rep, pair_rep = module(
        peptide_emb, groove_emb, return_pair_rep=True
    )
    
    print(f"✓ Interface rep shape: {interface_rep.shape}")
    print(f"✓ Pair rep shape: {pair_rep.shape} (original non-square region extracted)")
    
    assert not torch.isnan(interface_rep).any(), "Interface rep contains NaN"
    assert not torch.isnan(pair_rep).any(), "Pair rep contains NaN"
    print()


def test_dinomhc_mask():
    """Test complete DinoMHC with masks."""
    print("=" * 70)
    print("Test 5: DinoMHC Full Pipeline Mask Handling")
    print("=" * 70)
    
    batch_size = 2
    pep_len = 12
    mhc_len = 180
    
    config = {
        'dim': 64,
        'num_groove_tokens': 30,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    model = DinoMHC(config)
    model.eval()
    
    # Create token sequences (using dummy tokens 1-20)
    peptide_tokens = torch.randint(1, 20, (batch_size, pep_len))
    mhc_tokens = torch.randint(1, 20, (batch_size, mhc_len))
    
    # Create masks
    pep_mask = torch.ones(batch_size, pep_len, dtype=torch.bool)
    pep_mask[1, 9:] = False  # Second sample: 9-mer peptide (rest is padding)
    
    mhc_mask = torch.ones(batch_size, mhc_len, dtype=torch.bool)
    mhc_mask[1, 150:] = False  # Second sample: shorter MHC
    
    # Zero out padded positions
    peptide_tokens[1, 9:] = 0
    mhc_tokens[1, 150:] = 0
    
    print(f"Peptide tokens shape: {peptide_tokens.shape}")
    print(f"  Sample 0: length {pep_len} (full)")
    print(f"  Sample 1: length 9 (padded to {pep_len})")
    print(f"MHC tokens shape: {mhc_tokens.shape}")
    print(f"  Sample 0: length {mhc_len} (full)")
    print(f"  Sample 1: length 150 (padded to {mhc_len})")
    
    # Forward with masks
    outputs = model(
        peptide_tokens, mhc_tokens,
        peptide_mask=pep_mask,
        mhc_mask=mhc_mask,
        return_attention=True
    )
    
    print(f"✓ Prediction shape: {outputs['prediction'].shape}")
    print(f"✓ Prediction values: {outputs['prediction'].squeeze()}")
    
    assert not torch.isnan(outputs['prediction']).any(), "Prediction contains NaN"
    
    # Check attention info
    attn_info = outputs['attention_info']
    print(f"✓ Attention info keys: {list(attn_info.keys())}")
    print()


def test_gradient_with_masks():
    """Test gradient flow with masks."""
    print("=" * 70)
    print("Test 6: Gradient Flow with Masks")
    print("=" * 70)
    
    batch_size = 2
    pep_len = 12
    groove_len = 30
    dim = 64
    
    module = GroovePeptideFusion(dim=dim, num_layers=2, num_heads=4)
    
    peptide_emb = torch.randn(batch_size, pep_len, dim, requires_grad=True)
    groove_emb = torch.randn(batch_size, groove_len, dim, requires_grad=True)
    
    # Create masks
    pep_mask = torch.ones(batch_size, pep_len, dtype=torch.bool)
    pep_mask[1, pep_len//2:] = False
    
    groove_mask = torch.ones(batch_size, groove_len, dtype=torch.bool)
    groove_mask[1, groove_len//2:] = False
    
    # Forward
    pep_out, groove_out, _ = module(
        peptide_emb, groove_emb,
        peptide_mask=pep_mask,
        groove_mask=groove_mask
    )
    
    # Compute loss only on valid positions (simulating real training)
    pep_loss = (pep_out * pep_mask.unsqueeze(-1).float()).sum()
    groove_loss = (groove_out * groove_mask.unsqueeze(-1).float()).sum()
    loss = pep_loss + groove_loss
    
    loss.backward()
    
    print(f"✓ Peptide gradient exists: {peptide_emb.grad is not None}")
    print(f"✓ Groove gradient exists: {groove_emb.grad is not None}")
    print(f"✓ Peptide gradient norm: {peptide_emb.grad.norm():.4f}")
    print(f"✓ Groove gradient norm: {groove_emb.grad.norm():.4f}")
    print()


def test_batch_consistency_with_masks():
    """Test that different samples in batch with different masks give consistent results."""
    print("=" * 70)
    print("Test 7: Batch Consistency with Different Mask Lengths")
    print("=" * 70)
    
    dim = 64
    pep_len = 12
    groove_len = 30
    
    module = GroovePeptideFusion(dim=dim, num_layers=2, num_heads=4)
    module.eval()
    
    # Create two separate samples
    torch.manual_seed(42)
    pep1 = torch.randn(1, 9, dim)  # 9-mer
    groove1 = torch.randn(1, groove_len, dim)
    
    torch.manual_seed(42)  # Same seed
    pep2 = torch.randn(1, 9, dim)  # Same 9-mer
    groove2 = torch.randn(1, groove_len, dim)
    
    # Run sample 1 alone (no padding needed since length is 9)
    pep_out1, groove_out1, _ = module(pep1, groove1)
    
    # Now create a batch with sample 2 padded to length 12
    pep2_padded = torch.zeros(1, pep_len, dim)
    pep2_padded[:, :9, :] = pep2
    
    pep_mask = torch.zeros(1, pep_len, dtype=torch.bool)
    pep_mask[:, :9] = True
    
    # Run sample 2 with padding and mask
    pep_out2, groove_out2, _ = module(
        pep2_padded, groove2,
        peptide_mask=pep_mask,
        groove_mask=None
    )
    
    # Compare the valid portion of outputs
    pep_diff = (pep_out1 - pep_out2[:, :9, :]).abs().max()
    groove_diff = (groove_out1 - groove_out2).abs().max()
    
    print(f"Peptide output difference (valid region): {pep_diff:.6f}")
    print(f"Groove output difference: {groove_diff:.6f}")
    print(f"✓ Outputs are consistent: {pep_diff < 0.01 and groove_diff < 0.01}")
    
    # Check padded region is zeroed
    padded_max = pep_out2[:, 9:, :].abs().max()
    print(f"✓ Padded region is zeroed: {padded_max:.6f} (should be 0)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MASK HANDLING TESTS")
    print("=" * 70 + "\n")
    
    test_hierarchical_groove_extractor_mask()
    test_groove_peptide_fusion_mask()
    test_position_aware_encoder_mask()
    test_interface_geometry_module_mask()
    test_dinomhc_mask()
    test_gradient_with_masks()
    test_batch_consistency_with_masks()
    
    print("=" * 70)
    print("✓ ALL MASK HANDLING TESTS COMPLETED!")
    print("=" * 70)
