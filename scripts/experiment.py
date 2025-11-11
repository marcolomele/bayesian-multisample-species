#!/usr/bin/env python3
"""
HPYP species sampling experiment pipeline.

Usage:
    python experiment.py --config config.json
    python experiment.py --config config.json --resume
"""

import os
import sys
import pickle
import json
from datetime import datetime

from data_utils import load_data, split_data, prepare_input
from model_fitting import fit_independent_models, fit_dependent_model
from prediction import predict_independent, predict_dependent, compute_statistics, check_linearity
from output_utils import (
    create_independent_table, create_dependent_table, 
    create_parameter_table, create_comparison_table,
    save_linearity_check, print_summary
)


def checkpoint_path(output_dir: str, name: str) -> str:
    """Get checkpoint file path."""
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    return os.path.join(output_dir, 'checkpoints', f'{name}.pkl')


def save_checkpoint(output_dir: str, name: str, data: dict):
    """Save checkpoint."""
    path = checkpoint_path(output_dir, name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Checkpoint: {name}")


def load_checkpoint(output_dir: str, name: str):
    """Load checkpoint. Returns None if not found."""
    path = checkpoint_path(output_dir, name)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠ Corrupted checkpoint {name}: {e}")
            return None
    return None


def run_experiment(config: dict, resume: bool = False):
    """Run complete HPYP experiment from config."""
    
    print("\n" + "="*60)
    print("HPYP EXPERIMENT")
    print("="*60 + "\n")
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== STEP 1: DATA ====================
    print("STEP 1: Load and prepare data")
    df = load_data(
        config['data_path'],
        restaurants=config.get('restaurants'),
        max_customers=config.get('max_customers_per_restaurant'),
        restaurant_col=config.get('restaurant_col'),
        dish_col=config.get('dish_col'),
        count_col=config.get('count_col'),
        seed=config['seed']
    )
    
    restaurants = config.get('restaurants') or sorted(df['restaurant'].unique())
    fit_data, test_data = split_data(df, config['train_ratio'], config['seed'])
    data_dict, metadata = prepare_input(fit_data, restaurants)
    
    # ==================== STEP 2: FIT ====================
    print("\nSTEP 2: Fit models")
    checkpoint = load_checkpoint(output_dir, 'models') if resume else None
    
    if checkpoint:
        print("  [Resume] Loading from checkpoint...")
        models_ind = checkpoint['models_ind']
        model_dep = checkpoint['model_dep']
    else:
        models_ind = fit_independent_models(data_dict, metadata, config)
        model_dep = fit_dependent_model(data_dict, metadata, config)
        save_checkpoint(output_dir, 'models', {
            'models_ind': models_ind,
            'model_dep': model_dep
        })
    
    # ==================== STEP 3: PREDICT ====================
    print("\nSTEP 3: Generate predictions")
    checkpoint = load_checkpoint(output_dir, 'predictions') if resume else None
    
    num_threads = config.get('num_threads', 1)
    
    if checkpoint:
        print("  [Resume] Loading from checkpoint...")
        results_ind = checkpoint.get('results_ind')
        results_dep = checkpoint.get('results_dep')
    else:
        results_ind = None
        results_dep = None
    
    # Independent predictions
    if results_ind is None:
        print("\n  Running independent predictions...")
        results_ind = predict_independent(
            models_ind, data_dict, metadata, 
            config['m_values'], config['num_predict_iterations'],
            num_threads=num_threads
        )
        # Save immediately after independent predictions complete
        save_checkpoint(output_dir, 'predictions', {
            'results_ind': results_ind,
            'results_dep': None  # Placeholder
        })
        print("  ✓ Independent predictions saved to checkpoint")
    else:
        print("  [Resume] Using existing independent predictions")
    
    # Dependent predictions
    if results_dep is None:
        print("\n  Running dependent predictions...")
        results_dep = predict_dependent(
            model_dep, data_dict, metadata,
            config['m_values'], config['num_predict_iterations'],
            num_threads=num_threads
        )
        # Save after dependent predictions complete
        save_checkpoint(output_dir, 'predictions', {
            'results_ind': results_ind,
            'results_dep': results_dep
        })
        print("  ✓ Dependent predictions saved to checkpoint")
    else:
        print("  [Resume] Using existing dependent predictions")
    
    # ==================== STEP 4: STATISTICS ====================
    print("\nSTEP 4: Compute statistics")
    checkpoint = load_checkpoint(output_dir, 'statistics') if resume else None
    
    if checkpoint:
        print("  [Resume] Loading from checkpoint...")
        stats_ind = checkpoint['stats_ind']
        stats_dep = checkpoint['stats_dep']
        linearity_ind = checkpoint['linearity_ind']
        linearity_dep = checkpoint['linearity_dep']
    else:
        stats_ind = compute_statistics(results_ind, is_dependent=False)
        stats_dep = compute_statistics(results_dep, is_dependent=True)
        linearity_ind = check_linearity(stats_ind, config['m_values'])
        linearity_dep = check_linearity(stats_dep, config['m_values'])
        save_checkpoint(output_dir, 'statistics', {
            'stats_ind': stats_ind,
            'stats_dep': stats_dep,
            'linearity_ind': linearity_ind,
            'linearity_dep': linearity_dep
        })
    
    # ==================== STEP 5: OUTPUT ====================
    print("\nSTEP 5: Generate output")
    
    tables_dir = os.path.join(output_dir, 'tables')
    diag_dir = os.path.join(output_dir, 'diagnostics')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
    
    create_independent_table(stats_ind, config['m_values'], metadata,
                           os.path.join(tables_dir, 'independent_predictions.csv'))
    create_dependent_table(stats_dep, config['m_values'], metadata,
                         os.path.join(tables_dir, 'dependent_predictions.csv'))
    create_parameter_table(models_ind, model_dep, metadata,
                          os.path.join(tables_dir, 'parameter_estimates.csv'))
    create_comparison_table(stats_ind, stats_dep, config['m_values'], metadata,
                           os.path.join(tables_dir, 'model_comparison.csv'))
    save_linearity_check(linearity_ind, metadata,
                        os.path.join(diag_dir, 'linearity_independent.txt'))
    save_linearity_check(linearity_dep, metadata,
                        os.path.join(diag_dir, 'linearity_dependent.txt'))
    
    # Save full results
    with open(os.path.join(output_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump({
            'config': config, 'metadata': metadata,
            'models_ind': models_ind, 'model_dep': model_dep,
            'results_ind': results_ind, 'results_dep': results_dep,
            'stats_ind': stats_ind, 'stats_dep': stats_dep,
            'linearity_ind': linearity_ind, 'linearity_dep': linearity_dep,
            'timestamp': datetime.now().isoformat()
        }, f)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print_summary(stats_ind, stats_dep, metadata, m_value=1000)
    print(f"\n✓ Complete! Results in {output_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HPYP species sampling experiment pipeline"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration JSON file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved checkpoints")
    args = parser.parse_args()

    config_path = args.config
    resume = args.resume

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'data_path' not in config:
        print("Error: 'data_path' required in config")
        sys.exit(1)
    
    if not os.path.exists(config['data_path']):
        print(f"Error: Data file not found: {config['data_path']}")
        sys.exit(1)
    
    try:
        run_experiment(config, resume=resume)
        print("\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

