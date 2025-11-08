#!/usr/bin/env python3
"""
Main experiment script for HPYP species sampling analysis.

This script implements the complete experimental pipeline:
1. Load and split data
2. Fit independent and dependent HPYP models
3. Generate predictions for various sample sizes
4. Output comparison tables

Usage:
    python experiment.py [--config CONFIG_FILE] [--quick-test]
"""

import os
import sys
import argparse
import pickle
import json
from datetime import datetime

# Import our modules
from data_utils import load_names_data, load_names_data_stratified, split_data_random, prepare_hpyp_input
from model_fitting import (
    fit_independent_models, 
    fit_dependent_model,
    compare_parameter_estimates
)
from prediction import (
    predict_new_species_independent,
    predict_new_species_dependent,
    compute_statistics,
    check_linearity
)
from output_utils import (
    create_independent_table,
    create_dependent_table,
    create_parameter_table,
    create_comparison_table,
    save_linearity_check,
    print_summary
)


def run_experiment(data_path: str, config: dict) -> dict:
    """
    Main experimental pipeline.
    
    Args:
        data_path: Path to input data file
        config: Configuration dictionary with experimental settings
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "="*60)
    print("HIERARCHICAL PITMAN-YOR PROCESS EXPERIMENT")
    print("Bayesian Multi-Sample Species Discovery")
    print("="*60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        if key != 'output_dir':  # Skip long paths
            print(f"  {key}: {value}")
    print(f"  output_dir: {config['output_dir']}")
    print()
    
    # ========================================
    # 1. LOAD AND PREPARE DATA
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    year = config.get('year', None)
    states = config.get('states', ['CA', 'FL', 'NY', 'PA', 'TX'])
    max_obs_per_state = config.get('max_observations_per_state', None)
    
    if year:
        print(f"Filtering data to year {year}")
    if states:
        print(f"Filtering to states: {states}")
    if max_obs_per_state:
        print(f"Using stratified sampling: max {max_obs_per_state:,} per state")
        df = load_names_data_stratified(
            data_path,
            year=year,
            states=states,
            max_observations_per_state=max_obs_per_state,
            random_seed=config.get('seed', 42)
        )
    else:
        df = load_names_data(data_path, year=year)
    fit_data, predict_data = split_data_random(
        df, 
        train_ratio=config['train_ratio'], 
        random_seed=config['seed']
    )
    
    # Get states from config (already set above)
    fit_data_dict, metadata = prepare_hpyp_input(fit_data, states)
    
    # ========================================
    # 2. FIT MODELS
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: MODEL FITTING")
    print("="*60)
    
    # Fit independent models
    print("\n[2.1] Fitting independent models...")
    models_independent = fit_independent_models(fit_data_dict, metadata, config)
    
    # Fit dependent model
    print("\n[2.2] Fitting dependent model...")
    model_dependent = fit_dependent_model(fit_data_dict, metadata, config)
    
    # Compare parameters
    param_comparison = compare_parameter_estimates(
        models_independent, 
        model_dependent, 
        states
    )
    
    # ========================================
    # 3. GENERATE PREDICTIONS
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: PREDICTIONS")
    print("="*60)
    
    # Independent predictions
    print("\n[3.1] Generating independent predictions...")
    results_independent = predict_new_species_independent(
        models_independent,
        fit_data_dict,
        metadata,
        config['m_values'],
        num_iterations=config['num_predict_iterations'],
        verbose=True
    )
    
    # Dependent predictions
    print("\n[3.2] Generating dependent predictions...")
    results_dependent = predict_new_species_dependent(
        model_dependent,
        fit_data_dict,
        metadata,
        config['m_values'],
        num_iterations=config['num_predict_iterations'],
        verbose=True
    )
    
    # ========================================
    # 4. COMPUTE STATISTICS
    # ========================================
    print("\n" + "="*60)
    print("STEP 4: COMPUTING STATISTICS")
    print("="*60)
    
    print("\nComputing summary statistics...")
    stats_independent = compute_statistics(results_independent, is_dependent=False)
    stats_dependent = compute_statistics(results_dependent, is_dependent=True)
    
    print("Checking linearity (Theorem 1)...")
    linearity_independent = check_linearity(stats_independent, config['m_values'])
    linearity_dependent = check_linearity(stats_dependent, config['m_values'])
    
    # ========================================
    # 5. GENERATE OUTPUTS
    # ========================================
    print("\n" + "="*60)
    print("STEP 5: GENERATING OUTPUT TABLES")
    print("="*60)
    
    # Create output directories
    output_dir = config['output_dir']
    tables_dir = os.path.join(output_dir, 'tables')
    diagnostics_dir = os.path.join(output_dir, 'diagnostics')
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Create tables
    print("\nGenerating output tables...")
    
    df_ind = create_independent_table(
        stats_independent,
        config['m_values'],
        metadata,
        output_path=os.path.join(tables_dir, 'independent_predictions.csv')
    )
    
    df_dep = create_dependent_table(
        stats_dependent,
        config['m_values'],
        metadata,
        output_path=os.path.join(tables_dir, 'dependent_predictions.csv')
    )
    
    df_params = create_parameter_table(
        models_independent,
        model_dependent,
        metadata,
        output_path=os.path.join(tables_dir, 'parameter_estimates.csv')
    )
    
    df_comparison = create_comparison_table(
        stats_independent,
        stats_dependent,
        config['m_values'],
        metadata,
        output_path=os.path.join(tables_dir, 'model_comparison.csv')
    )
    
    # Save linearity checks
    save_linearity_check(
        linearity_independent,
        metadata,
        output_path=os.path.join(diagnostics_dir, 'linearity_independent.txt')
    )
    
    save_linearity_check(
        linearity_dependent,
        metadata,
        output_path=os.path.join(diagnostics_dir, 'linearity_dependent.txt')
    )
    
    # ========================================
    # 6. SAVE COMPLETE RESULTS
    # ========================================
    print("\nSaving complete results...")
    
    results = {
        'config': config,
        'metadata': metadata,
        'models_independent': models_independent,
        'model_dependent': model_dependent,
        'results_independent': results_independent,
        'results_dependent': results_dependent,
        'stats_independent': stats_independent,
        'stats_dependent': stats_dependent,
        'linearity_independent': linearity_independent,
        'linearity_dependent': linearity_dependent,
        'param_comparison': param_comparison,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save as pickle
    with open(os.path.join(output_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Complete results saved to {os.path.join(output_dir, 'full_results.pkl')}")
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        # Convert non-serializable items
        config_serializable = {k: v for k, v in config.items() 
                              if not callable(v)}
        json.dump(config_serializable, f, indent=2)
    print(f"✓ Configuration saved to {os.path.join(output_dir, 'config.json')}")
    
    # ========================================
    # 7. PRINT SUMMARY
    # ========================================
    print_summary(stats_independent, stats_dependent, metadata, m_value=1000)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nOutput files:")
    print(f"  - tables/independent_predictions.csv")
    print(f"  - tables/dependent_predictions.csv")
    print(f"  - tables/parameter_estimates.csv")
    print(f"  - tables/model_comparison.csv")
    print(f"  - diagnostics/linearity_independent.txt")
    print(f"  - diagnostics/linearity_dependent.txt")
    print(f"  - full_results.pkl")
    print(f"  - config.json")
    
    return results


def create_default_config(quick_test=False, year=2024):
    """Create default configuration."""
    if quick_test:
        # Quick test configuration with minimal iterations and 30% of data
        return {
            'train_ratio': 0.8,
            'seed': 42,
            'year': year,
            'states': ['CA', 'TX'],  # 2 states like original paper
            'max_observations_per_state': 30000,  # 30% of 100k = 30k per state
            'num_fit_iterations': 100,
            'burn_in': 50,
            'num_predict_iterations': 100,
            'm_values': [200, 500, 1000, 1500, 2000],  # 5 m values
            'd_0': 0.5,
            'theta_0': 1000.0,
            'd_j': 0.5,
            'theta_j': 1000.0,
            'update_params': True,
            'verbose': True,
            'output_dir': './results_quick_test'
        }
    else:
        # Full configuration with hybrid approach
        return {
            'train_ratio': 0.8,
            'seed': 42,
            'year': year,
            'states': ['CA', 'TX'],  # 2 states like original paper
            'max_observations_per_state': 100000,  # Cap at 100k per state
            'num_fit_iterations': 1000,
            'burn_in': 500,
            'num_predict_iterations': 1000,
            'm_values': [200, 500, 1000, 1500, 2000],  # 5 m values
            'd_0': 0.5,
            'theta_0': 1000.0,
            'd_j': 0.5,
            'theta_j': 1000.0,
            'update_params': True,
            'verbose': True,
            'output_dir': './results'
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='HPYP Species Sampling Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with minimal iterations (2024 data)
  python experiment.py --quick-test
  
  # Full experiment with 2024 data (default)
  python experiment.py
  
  # Use different year
  python experiment.py --year 2023
  
  # Use all years (no filter)
  python experiment.py --year None
  
  # Custom data path
  python experiment.py --data /path/to/data.csv
  
  # Custom configuration
  python experiment.py --config my_config.json
        """
    )
    
    parser.add_argument(
        '--data',
        default='../data/namesbystate_subset.csv',
        help='Path to input data file (default: ../data/namesbystate_subset.csv)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to JSON configuration file (optional)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal iterations'
    )
    
    parser.add_argument(
        '--year',
        type=lambda x: None if x.lower() == 'none' or x == '0' else int(x),
        default=2024,
        help='Year to filter data (default: 2024). Use --year None or --year 0 for all years.'
    )
    
    parser.add_argument(
        '--output',
        help='Output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config(quick_test=args.quick_test, year=args.year)
        if args.quick_test:
            print("Running in QUICK TEST mode (reduced iterations)")
    
    # Override year from CLI (takes precedence over config file)
    config['year'] = args.year
    
    # Override output directory if specified
    if args.output:
        config['output_dir'] = args.output
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    try:
        # Run experiment
        results = run_experiment(args.data, config)
        
        print("\n✓ Experiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

