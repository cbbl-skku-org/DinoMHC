#!/usr/bin/env python
"""
Debug test to identify which component causes the mask mismatch.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/duongtt/Workdir/MHC/digermhc/src')
from model import (
    DinoMHC, 
    HierarchicalGrooveExtractor,
    PositionAwarePeptideEncoder,
    GroovePeptideFusion,
    InterfaceGeometryModule
)

torch.manual_seed(42)

def test_step_by_step():
    """Step-by-step comparison to identify the issue."""
    print("=" * 70)
    print("STEP-BY-STEP MASK DEBUGGING")
    print("=" * 70)
    
    # Config
    dim = 64
    pep_valid = 9
    pep_padding = 5
    pep_full = pep_valid + pep_padding
    mhc_valid = 150
    mhc_padding = 30
    mhc_full = mhc_valid + mhc_padding
    num_groove = 50
    
    # Create modules
    groove_extractor = HierarchicalGrooveExtractor(dim=dim, num_groove_tokens=num_groove).eval()
    position_encoder = PositionAwarePeptideEncoder(dim=dim, max_peptide_length=15).eval()
    fusion = GroovePeptideFusion(dim=dim, num_layers=2).eval()
    interface_geom = InterfaceGeometryModule(dim=dim, num_blocks=2).eval()
    
    # --- Method 1: Padded with mask ---
    torch.manual_seed(42)
    pep_emb_full = torch.randn(1, pep_full, dim)
    mhc_emb_full = torch.randn(1, mhc_full, dim)
    peptide_mask = torch.zeros(1, pep_full, dtype=torch.bool)
    peptide_mask[:, :pep_valid] = True
    mhc_mask = torch.zeros(1, mhc_full, dtype=torch.bool)
    mhc_mask[:, :mhc_valid] = True
    
    # --- Method 2: Sliced ---
    pep_emb_sliced = pep_emb_full[:, :pep_valid, :].clone()
    mhc_emb_sliced = mhc_emb_full[:, :mhc_valid, :].clone()
    
    # STEP 1: Groove Extraction
    print("\n=== STEP 1: Groove Extraction ===")
    with torch.no_grad():
        groove1, _ = groove_extractor(mhc_emb_full, pep_emb_full, mhc_mask=mhc_mask, peptide_mask=peptide_mask)
        groove2, _ = groove_extractor(mhc_emb_sliced, pep_emb_sliced)
    
    diff1 = (groove1 - groove2).abs().max().item()
    print(f"Groove diff: {diff1:.10f}")
    if diff1 < 1e-5:
        print("✓ PASS")
    else:
        print("✗ FAIL")
    
    # STEP 2: Position Encoding
    print("\n=== STEP 2: Position Encoding ===")
    with torch.no_grad():
        pep_enhanced1, _ = position_encoder(pep_emb_full, peptide_mask=peptide_mask)
        pep_enhanced2, _ = position_encoder(pep_emb_sliced)
    
    # Compare only valid portion
    diff2 = (pep_enhanced1[:, :pep_valid, :] - pep_enhanced2).abs().max().item()
    print(f"Position encoder valid portion diff: {diff2:.10f}")
    if diff2 < 1e-5:
        print("✓ PASS")
    else:
        print("✗ FAIL")
    
    # Check if padding portion is zero
    padding_max = pep_enhanced1[:, pep_valid:, :].abs().max().item()
    print(f"Padding portion max (should be 0): {padding_max:.10f}")
    
    # STEP 3: Fusion
    print("\n=== STEP 3: Groove-Peptide Fusion ===")
    with torch.no_grad():
        pep_fused1, groove_fused1, _ = fusion(pep_enhanced1, groove1, peptide_mask=peptide_mask, groove_mask=None)
        pep_fused2, groove_fused2, _ = fusion(pep_enhanced2, groove2)
    
    # Compare valid portions
    diff_pep = (pep_fused1[:, :pep_valid, :] - pep_fused2).abs().max().item()
    diff_groove = (groove_fused1 - groove_fused2).abs().max().item()
    print(f"Fused peptide valid portion diff: {diff_pep:.10f}")
    print(f"Fused groove diff: {diff_groove:.10f}")
    if diff_pep < 1e-5 and diff_groove < 1e-5:
        print("✓ PASS")
    else:
        print("✗ FAIL")
    
    # Check if peptide padding portion is zero
    pep_padding_max = pep_fused1[:, pep_valid:, :].abs().max().item()
    print(f"Peptide fused padding portion max (should be 0): {pep_padding_max:.10f}")
    
    # STEP 4: Interface Geometry
    print("\n=== STEP 4: Interface Geometry ===")
    print("Note: InterfaceGeometryModule handles internal square padding")
    
    with torch.no_grad():
        # For padded version - need to pass only valid peptide portion
        # But wait - fusion returns full length including padding!
        # Let's test both approaches
        
        # Approach A: Pass full fused with padding (masked out as zeros)
        interface1, pair1 = interface_geom(pep_fused1[:, :pep_valid, :], groove_fused1, return_pair_rep=True)
        
        # Approach B: Pass sliced
        interface2, pair2 = interface_geom(pep_fused2, groove_fused2, return_pair_rep=True)
    
    diff_interface = (interface1 - interface2).abs().max().item()
    diff_pair = (pair1 - pair2).abs().max().item()
    print(f"Interface rep diff: {diff_interface:.10f}")
    print(f"Pair rep diff: {diff_pair:.10f}")
    if diff_interface < 1e-5 and diff_pair < 1e-5:
        print("✓ PASS")
    else:
        print("✗ FAIL - let's debug further")
        
        # The inputs should be identical at this point
        print(f"\nInput pep_fused1[:, :pep_valid, :] vs pep_fused2 diff: {(pep_fused1[:, :pep_valid, :] - pep_fused2).abs().max().item():.10f}")
        print(f"Input groove_fused1 vs groove_fused2 diff: {(groove_fused1 - groove_fused2).abs().max().item():.10f}")


def test_full_model_step_by_step():
    """Test the full model but intercept intermediate outputs."""
    print("\n" + "=" * 70)
    print("FULL MODEL STEP-BY-STEP")
    print("=" * 70)
    
    # Config
    config = {
        'dim': 64,
        'num_groove_tokens': 50,
        'num_fusion_layers': 2,
        'num_geometry_blocks': 2,
        'num_heads': 4,
        'dropout': 0.0,  # No dropout for determinism
        'max_peptide_length': 15,
        'task_head': 'presentation'
    }
    
    pep_valid = 9
    pep_padding = 5
    pep_full = pep_valid + pep_padding
    mhc_valid = 150
    mhc_padding = 30
    mhc_full = mhc_valid + mhc_padding
    
    model = DinoMHC(config).eval()
    
    # Create tokens
    torch.manual_seed(42)
    pep_tokens_full = torch.randint(1, 20, (1, pep_full))
    mhc_tokens_full = torch.randint(1, 20, (1, mhc_full))
    
    pep_tokens_sliced = pep_tokens_full[:, :pep_valid].clone()
    mhc_tokens_sliced = mhc_tokens_full[:, :mhc_valid].clone()
    
    peptide_mask = torch.zeros(1, pep_full, dtype=torch.bool)
    peptide_mask[:, :pep_valid] = True
    mhc_mask = torch.zeros(1, mhc_full, dtype=torch.bool)
    mhc_mask[:, :mhc_valid] = True
    
    with torch.no_grad():
        # Manual forward pass for Method 1 (padded with mask)
        print("\n--- Method 1: Padded with mask ---")
        pep_emb1, mhc_emb1, pep_mask_out1, mhc_mask_out1 = model.encode_sequences(pep_tokens_full, mhc_tokens_full)
        print(f"pep_emb1 shape: {pep_emb1.shape}")
        print(f"mhc_emb1 shape: {mhc_emb1.shape}")
        
        groove1, _ = model.groove_extractor(mhc_emb1, pep_emb1, mhc_mask=mhc_mask, peptide_mask=peptide_mask)
        print(f"groove1 shape: {groove1.shape}")
        
        pep_enhanced1, _ = model.position_encoder(pep_emb1, peptide_mask=peptide_mask)
        print(f"pep_enhanced1 shape: {pep_enhanced1.shape}")
        print(f"pep_enhanced1 padding region max: {pep_enhanced1[:, pep_valid:, :].abs().max().item():.10f}")
        
        pep_fused1, groove_fused1, _ = model.groove_peptide_fusion(
            pep_enhanced1, groove1, peptide_mask=peptide_mask, groove_mask=None
        )
        print(f"pep_fused1 shape: {pep_fused1.shape}")
        print(f"pep_fused1 padding region max: {pep_fused1[:, pep_valid:, :].abs().max().item():.10f}")
        
        # THE KEY ISSUE: InterfaceGeometryModule gets the full pep_fused1 including padding!
        print(f"\n*** InterfaceGeometryModule input: ***")
        print(f"    peptide_fused (full): {pep_fused1.shape} - includes {pep_padding} padding positions")
        
        interface1, pair1 = model.interface_geometry(pep_fused1, groove_fused1, return_pair_rep=True)
        print(f"interface1 shape: {interface1.shape}")
        
        # --- Method 2: Sliced ---
        print("\n--- Method 2: Sliced ---")
        pep_emb2, mhc_emb2, pep_mask_out2, mhc_mask_out2 = model.encode_sequences(pep_tokens_sliced, mhc_tokens_sliced)
        print(f"pep_emb2 shape: {pep_emb2.shape}")
        print(f"mhc_emb2 shape: {mhc_emb2.shape}")
        
        groove2, _ = model.groove_extractor(mhc_emb2, pep_emb2)
        print(f"groove2 shape: {groove2.shape}")
        
        pep_enhanced2, _ = model.position_encoder(pep_emb2)
        print(f"pep_enhanced2 shape: {pep_enhanced2.shape}")
        
        pep_fused2, groove_fused2, _ = model.groove_peptide_fusion(pep_enhanced2, groove2)
        print(f"pep_fused2 shape: {pep_fused2.shape}")
        
        interface2, pair2 = model.interface_geometry(pep_fused2, groove_fused2, return_pair_rep=True)
        print(f"interface2 shape: {interface2.shape}")
        
        # Compare
        print("\n--- Comparison ---")
        print(f"groove diff: {(groove1 - groove2).abs().max().item():.10f}")
        print(f"pep_enhanced valid diff: {(pep_enhanced1[:, :pep_valid, :] - pep_enhanced2).abs().max().item():.10f}")
        print(f"pep_fused valid diff: {(pep_fused1[:, :pep_valid, :] - pep_fused2).abs().max().item():.10f}")
        print(f"groove_fused diff: {(groove_fused1 - groove_fused2).abs().max().item():.10f}")
        print(f"interface diff: {(interface1 - interface2).abs().max().item():.10f}")


if __name__ == '__main__':
    test_step_by_step()
    test_full_model_step_by_step()
