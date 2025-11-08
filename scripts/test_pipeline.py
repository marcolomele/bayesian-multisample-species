#!/usr/bin/env python3
"""
Simple test script to validate the pipeline components.
"""

import os
import sys

def test_data_loading():
    """Test data loading and splitting."""
    print("Testing data loading...")
    from data_utils import load_names_data, split_data_random, prepare_hpyp_input
    
    df = load_names_data('../data/namesbystate_test.csv')
    assert len(df) > 0, "Data loading failed"
    
    fit_data, predict_data = split_data_random(df, train_ratio=0.8, random_seed=42)
    assert len(fit_data) > 0, "Data splitting failed"
    assert len(predict_data) > 0, "Prediction data is empty"
    
    states = ['CA', 'FL', 'NY', 'PA', 'TX']
    data_dict, metadata = prepare_hpyp_input(fit_data, states)
    assert len(data_dict) > 0, "HPYP input preparation failed"
    
    print("✓ Data loading tests passed!")
    return data_dict, metadata


def test_model_creation():
    """Test HPYP model creation."""
    print("\nTesting model creation...")
    from pitmanyor import HierarchicalPitmanYorProcess
    
    model = HierarchicalPitmanYorProcess(
        d_0=0.5,
        theta_0=100.0,
        d_j=0.5,
        theta_j=100.0,
        num_groups=3
    )
    assert model is not None, "Model creation failed"
    
    print("✓ Model creation tests passed!")
    return model


def test_model_fitting():
    """Test model fitting with minimal data."""
    print("\nTesting model fitting...")
    from pitmanyor import HierarchicalPitmanYorProcess
    
    # Create minimal test data
    test_data = {
        0: ['A', 'B', 'A', 'C', 'B', 'A'],
        1: ['X', 'Y', 'X', 'Y', 'Z']
    }
    
    model = HierarchicalPitmanYorProcess(
        d_0=0.5,
        theta_0=10.0,
        d_j=0.5,
        theta_j=10.0,
        num_groups=2
    )
    
    # Fit with minimal iterations
    posterior = model.fit_from_data(
        test_data,
        num_iterations=10,
        burn_in=5,
        update_params=False,
        verbose=False
    )
    
    assert len(posterior['theta_0']) == 5, "Posterior sampling failed"
    
    print("✓ Model fitting tests passed!")


def test_prediction():
    """Test prediction functionality."""
    print("\nTesting prediction...")
    from pitmanyor import HierarchicalPitmanYorProcess
    
    # Create and fit minimal model
    test_data = {
        0: ['A', 'B', 'A', 'C'] * 10
    }
    
    model = HierarchicalPitmanYorProcess(
        d_0=0.5,
        theta_0=10.0,
        d_j=0.5,
        theta_j=10.0,
        num_groups=1
    )
    
    model.fit_from_data(
        test_data,
        num_iterations=10,
        burn_in=5,
        update_params=False,
        verbose=False
    )
    
    # Generate predictions
    observed = set(['A', 'B', 'C'])
    samples, counts = model.sample_predictive(
        group_id=0,
        num_samples=10,
        observed_dishes=observed
    )
    
    assert len(samples) == 10, "Prediction failed"
    
    print("✓ Prediction tests passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("PIPELINE COMPONENT TESTS")
    print("="*60)
    
    try:
        data_dict, metadata = test_data_loading()
        test_model_creation()
        test_model_fitting()
        test_prediction()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe pipeline is ready to use.")
        print("Run: python experiment.py --quick-test")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

