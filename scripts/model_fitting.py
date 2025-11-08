"""
Model fitting utilities for HPYP experiments.
Handles both independent and dependent model configurations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from pitmanyor import HierarchicalPitmanYorProcess
from data_utils import expand_customers, expand_observations  # expand_observations is legacy alias


def fit_independent_models(
    data_dict: Dict[int, List[Tuple[str, int]]],
    metadata: Dict[str, Any],
    config: Dict[str, Any]
) -> List[HierarchicalPitmanYorProcess]:
    """
    Fit independent HPYP models (one per group, no sharing).
    
    Each state is modeled independently with its own parameters.
    
    Args:
        data_dict: {group_id: [(name, count), ...]}
        metadata: Metadata from prepare_hpyp_input
        config: Configuration dictionary with:
            - num_fit_iterations: Number of Gibbs iterations
            - burn_in: Number of burn-in iterations
            - d_0, theta_0, d_j, theta_j: Initial hyperparameters
            
    Returns:
        List of fitted HPYP models (one per group)
    """
    print("\n" + "="*60)
    print("FITTING INDEPENDENT MODELS")
    print("="*60)
    
    num_groups = metadata['num_groups']
    models = []
    
    for group_id in range(num_groups):
        # Skip if this group has no data
        if group_id not in data_dict:
            print(f"\nSkipping group {group_id} ({metadata['group_to_state'][group_id]}) - no data")
            models.append(None)
            continue
            
        state = metadata['group_to_state'][group_id]
        print(f"\nFitting model for group {group_id} ({state})...")
        
        # Create independent model (num_groups=1 means no sharing)
        model = HierarchicalPitmanYorProcess(
            d_0=config.get('d_0', 0.5),
            theta_0=config.get('theta_0', 1000.0),
            d_j=config.get('d_j', 0.5),
            theta_j=config.get('theta_j', 1000.0),
            num_groups=1  # Single group = independent
        )
        
        # Prepare data for this group only (map to group_id=0 since it's a single group)
        # Expand customers: convert (dish, count) to list of dishes
        group_data = {0: expand_customers({group_id: data_dict[group_id]})[group_id]}
        
        # Fit the model
        posterior_samples = model.fit_from_data(
            group_data,
            num_iterations=config.get('num_fit_iterations', 1000),
            burn_in=config.get('burn_in', 500),
            update_params=config.get('update_params', True),
            verbose=config.get('verbose', True)
        )
        
        # Store posterior samples in the model for later use
        model.posterior_samples = posterior_samples
        
        models.append(model)
        
        print(f"✓ Completed fitting for group {group_id} ({state})")
    
    print("\n✓ All independent models fitted successfully!")
    return models


def fit_dependent_model(
    data_dict: Dict[int, List[Tuple[str, int]]],
    metadata: Dict[str, Any],
    config: Dict[str, Any]
) -> HierarchicalPitmanYorProcess:
    """
    Fit a dependent HPYP model (all groups share base distribution G_0).
    
    This allows borrowing of strength across states.
    
    Args:
        data_dict: {group_id: [(name, count), ...]}
        metadata: Metadata from prepare_hpyp_input
        config: Configuration dictionary
        
    Returns:
        Fitted HPYP model with multiple groups
    """
    print("\n" + "="*60)
    print("FITTING DEPENDENT MODEL")
    print("="*60)
    
    num_groups = metadata['num_groups']
    
    # Create dependent model (num_groups > 1 means sharing via G_0)
    model = HierarchicalPitmanYorProcess(
        d_0=config.get('d_0', 0.5),
        theta_0=config.get('theta_0', 1000.0),
        d_j=config.get('d_j', 0.5),
        theta_j=config.get('theta_j', 1000.0),
        num_groups=num_groups  # Multiple groups sharing base
    )
    
    # Expand customers to individual dishes
    expanded_data = expand_customers(data_dict)
    
    # Fit the model
    print(f"Fitting model with {num_groups} groups sharing base distribution...")
    posterior_samples = model.fit_from_data(
        expanded_data,
        num_iterations=config.get('num_fit_iterations', 1000),
        burn_in=config.get('burn_in', 500),
        update_params=config.get('update_params', True),
        verbose=config.get('verbose', True)
    )
    
    # Store posterior samples
    model.posterior_samples = posterior_samples
    
    print("\n✓ Dependent model fitted successfully!")
    return model


def get_parameter_estimates(model: HierarchicalPitmanYorProcess) -> Dict[str, float]:
    """
    Extract posterior mean parameter estimates from a fitted model.
    
    Args:
        model: Fitted HPYP model with posterior_samples attribute
        
    Returns:
        Dictionary with parameter estimates
    """
    if not hasattr(model, 'posterior_samples'):
        raise ValueError("Model has not been fitted yet")
    
    samples = model.posterior_samples
    
    estimates = {
        'theta_0_mean': np.mean(samples['theta_0']),
        'theta_0_std': np.std(samples['theta_0']),
        'theta_j_mean': np.mean(samples['theta_j']),
        'theta_j_std': np.std(samples['theta_j']),
        'd_0_mean': np.mean(samples['d_0']),
        'd_0_std': np.std(samples['d_0']),
        'd_j_mean': np.mean(samples['d_j']),
        'd_j_std': np.std(samples['d_j']),
        'num_base_tables_mean': np.mean(samples['num_base_tables']),
        'num_unique_dishes_mean': np.mean(samples['num_unique_dishes'])
    }
    
    return estimates


def compare_parameter_estimates(
    independent_models: List[HierarchicalPitmanYorProcess],
    dependent_model: HierarchicalPitmanYorProcess,
    states: List[str]
) -> Dict[str, Any]:
    """
    Compare parameter estimates between independent and dependent models.
    
    Args:
        independent_models: List of independent models
        dependent_model: Single dependent model
        states: List of state names
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'independent': {},
        'dependent': get_parameter_estimates(dependent_model)
    }
    
    # Get estimates for each independent model
    for i, (model, state) in enumerate(zip(independent_models, states)):
        comparison['independent'][state] = get_parameter_estimates(model)
    
    return comparison


if __name__ == "__main__":
    # Test model fitting
    import os
    from data_utils import load_names_data, split_data_random, prepare_hpyp_input
    
    print("Testing model fitting pipeline...")
    
    data_path = os.path.join('..', 'data', 'namesbystate_subset.csv')
    
    # Load and prepare data
    df = load_names_data(data_path)
    fit_data, _ = split_data_random(df, train_ratio=0.8, random_seed=42)
    
    states = ['CA', 'FL', 'NY', 'PA', 'TX']
    data_dict, metadata = prepare_hpyp_input(fit_data, states)
    
    # Test config with small iterations
    config = {
        'num_fit_iterations': 100,
        'burn_in': 50,
        'd_0': 0.5,
        'theta_0': 1000.0,
        'd_j': 0.5,
        'theta_j': 1000.0,
        'update_params': True,
        'verbose': True
    }
    
    # Fit independent models (just first state for testing)
    print("\nTesting independent model fitting (first state only)...")
    test_data = {0: data_dict[0]}
    test_metadata = {**metadata, 'num_groups': 1}
    models_ind = fit_independent_models(test_data, test_metadata, config)
    
    print("\n✓ Model fitting test successful!")

