"""
Prediction algorithms for species sampling using HPYP.
Implements both independent and dependent prediction strategies.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
from tqdm import tqdm
from pitmanyor import HierarchicalPitmanYorProcess
import copy


def predict_new_species_independent(
    models: List[HierarchicalPitmanYorProcess],
    fit_data_dict: Dict[int, List[Tuple[str, int]]],
    metadata: Dict[str, Any],
    m_values: List[int],
    num_iterations: int = 1000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate predictions using independent models.
    
    Each state is predicted independently without borrowing strength.
    
    Args:
        models: List of fitted independent HPYP models
        fit_data_dict: Training data {group_id: [(name, count), ...]}
        metadata: Metadata dictionary
        m_values: List of prediction sizes to evaluate
        num_iterations: Number of prediction iterations
        verbose: Whether to show progress
        
    Returns:
        Dictionary with prediction results for each group and m value
    """
    if verbose:
        print("\n" + "="*60)
        print("INDEPENDENT PREDICTION")
        print("="*60)
    
    num_groups = len(models)
    
    # Get observed names for each group
    observed_names_per_group = {}
    for group_id in range(num_groups):
        observed_names_per_group[group_id] = set(name for name, _ in fit_data_dict[group_id])
    
    # Store results: results[group_id][m] = list of L^0 values across iterations
    results = {group_id: {m: [] for m in m_values} for group_id in range(num_groups)}
    
    # Run predictions for each group independently
    for group_id, model in enumerate(models):
        state = metadata['group_to_state'][group_id]
        
        if verbose:
            print(f"\nPredicting for group {group_id} ({state})...")
        
        observed_names = observed_names_per_group[group_id]
        
        # Iterate over prediction sizes
        for m in tqdm(m_values, desc=f"Group {group_id} prediction", disable=not verbose):
            # Run multiple prediction iterations
            for iteration in range(num_iterations):
                # Copy model state for this iteration
                model_copy = copy.deepcopy(model)
                
                # Generate m new samples
                # Note: model has group_id=0 since it's independent
                samples, _ = model_copy.sample_predictive(
                    group_id=0,  # Independent models have single group
                    num_samples=m,
                    observed_dishes=observed_names
                )
                
                # Count new species (not in training data)
                L_0 = sum(1 for sample in samples if sample not in observed_names)
                results[group_id][m].append(L_0)
    
    if verbose:
        print("\n✓ Independent predictions complete!")
    
    return results


def predict_new_species_dependent(
    model: HierarchicalPitmanYorProcess,
    fit_data_dict: Dict[int, List[Tuple[str, int]]],
    metadata: Dict[str, Any],
    m_values: List[int],
    num_iterations: int = 1000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate predictions using dependent model.
    
    All states share the base distribution, enabling borrowing of strength.
    Tracks multiple types of new species:
    - L^(0,0): New to all groups
    - L^(i,j): Seen in group j but new to group i
    - L^0: Total new to group i
    
    Args:
        model: Fitted dependent HPYP model
        fit_data_dict: Training data {group_id: [(name, count), ...]}
        metadata: Metadata dictionary
        m_values: List of prediction sizes to evaluate
        num_iterations: Number of prediction iterations
        verbose: Whether to show progress
        
    Returns:
        Dictionary with detailed prediction results
    """
    if verbose:
        print("\n" + "="*60)
        print("DEPENDENT PREDICTION")
        print("="*60)
    
    num_groups = metadata['num_groups']
    
    # Get observed names for each group
    observed_names_per_group = {}
    all_observed_names = set()
    for group_id in range(num_groups):
        observed = set(name for name, _ in fit_data_dict[group_id])
        observed_names_per_group[group_id] = observed
        all_observed_names.update(observed)
    
    # Initialize result storage
    results = {
        group_id: {
            m: {
                'L_0_0': [],  # New to all groups
                'L_0': [],     # Total new to this group
                'L_from_other': {other: [] for other in range(num_groups) if other != group_id}
            }
            for m in m_values
        }
        for group_id in range(num_groups)
    }
    
    # Run predictions
    if verbose:
        print(f"\nRunning {num_iterations} iterations for each m value...")
    
    for m in tqdm(m_values, desc="Prediction", disable=not verbose):
        for iteration in range(num_iterations):
            # Copy model state for this iteration
            model_copy = copy.deepcopy(model)
            
            # Generate predictions for all groups
            predictions_per_group = {}
            for group_id in range(num_groups):
                samples, _ = model_copy.sample_predictive(
                    group_id=group_id,
                    num_samples=m,
                    observed_dishes=all_observed_names
                )
                predictions_per_group[group_id] = samples
            
            # Analyze predictions for each group
            for group_id in range(num_groups):
                samples = predictions_per_group[group_id]
                observed_in_group = observed_names_per_group[group_id]
                
                # Count different types of new species
                L_0_0 = 0  # New to all groups
                L_0 = 0    # New to this group
                L_from_other = {other: 0 for other in range(num_groups) if other != group_id}
                
                for sample in samples:
                    if sample not in observed_in_group:
                        L_0 += 1
                        
                        # Check if it's new to all groups
                        is_new_to_all = True
                        for other_group in range(num_groups):
                            if other_group != group_id:
                                if sample in observed_names_per_group[other_group]:
                                    is_new_to_all = False
                                    L_from_other[other_group] += 1
                                    break
                        
                        if is_new_to_all:
                            L_0_0 += 1
                
                # Store results
                results[group_id][m]['L_0_0'].append(L_0_0)
                results[group_id][m]['L_0'].append(L_0)
                for other in L_from_other:
                    results[group_id][m]['L_from_other'][other].append(L_from_other[other])
    
    if verbose:
        print("\n✓ Dependent predictions complete!")
    
    return results


def compute_statistics(
    prediction_results: Dict[str, Any],
    alpha: float = 0.05,
    is_dependent: bool = False
) -> Dict[str, Any]:
    """
    Compute summary statistics from prediction results.
    
    Args:
        prediction_results: Raw prediction results
        alpha: Significance level for HPD intervals (default 0.05 for 95%)
        is_dependent: Whether results are from dependent model
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {}
    
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    for group_id, m_dict in prediction_results.items():
        stats[group_id] = {}
        
        for m, values in m_dict.items():
            if is_dependent:
                # Dependent model has structured results
                stats[group_id][m] = {
                    'L_0_0_mean': np.mean(values['L_0_0']),
                    'L_0_0_hpd': (
                        np.percentile(values['L_0_0'], lower_percentile),
                        np.percentile(values['L_0_0'], upper_percentile)
                    ),
                    'L_0_mean': np.mean(values['L_0']),
                    'L_0_hpd': (
                        np.percentile(values['L_0'], lower_percentile),
                        np.percentile(values['L_0'], upper_percentile)
                    ),
                    'L_from_other': {}
                }
                
                for other_group, other_values in values['L_from_other'].items():
                    stats[group_id][m]['L_from_other'][other_group] = {
                        'mean': np.mean(other_values),
                        'hpd': (
                            np.percentile(other_values, lower_percentile),
                            np.percentile(other_values, upper_percentile)
                        )
                    }
            else:
                # Independent model has simple list of L_0 values
                stats[group_id][m] = {
                    'L_0_mean': np.mean(values),
                    'L_0_hpd': (
                        np.percentile(values, lower_percentile),
                        np.percentile(values, upper_percentile)
                    ),
                    'L_0_std': np.std(values)
                }
    
    return stats


def check_linearity(stats: Dict[str, Any], m_values: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Check if L_m is linear in m (Theorem 1 verification).
    
    Args:
        stats: Summary statistics from compute_statistics
        m_values: List of m values used
        
    Returns:
        Dictionary with linearity metrics per group
    """
    from scipy import stats as scipy_stats
    
    linearity_results = {}
    
    for group_id in stats.keys():
        L_means = [stats[group_id][m]['L_0_mean'] for m in m_values]
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(m_values, L_means)
        
        linearity_results[group_id] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err
        }
    
    return linearity_results


if __name__ == "__main__":
    # Test prediction pipeline
    print("Prediction module loaded successfully!")
    print("Run experiment.py to test the complete pipeline.")

