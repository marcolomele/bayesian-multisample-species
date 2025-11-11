"""
Prediction algorithms for species sampling using HPYP.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from pitmanyor import HierarchicalPitmanYorProcess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy


# Module-level helper functions for ProcessPoolExecutor (must be picklable)
def _predict_iteration_independent(args):
    """Helper function for parallel independent prediction."""
    group_id, model, m, observed_names = args
    # Use efficient copy method instead of deepcopy
    model_copy = model.copy()
    samples, _ = model_copy.sample_predictive(
        group_id=0,  # Independent models have single group
        num_samples=m,
        observed_dishes=observed_names
    )
    return sum(1 for sample in samples if sample not in observed_names)


def _predict_iteration_dependent(args):
    """Helper function for parallel dependent prediction."""
    model, m, num_groups, all_observed_names, observed_names_per_group = args
    # Use efficient copy method instead of deepcopy
    model_copy = model.copy()
    
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
    iteration_results = {}
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
        
        iteration_results[group_id] = {
            'L_0_0': L_0_0,
            'L_0': L_0,
            'L_from_other': L_from_other
        }
    
    return iteration_results


def predict_independent(
    models: List[HierarchicalPitmanYorProcess],
    fit_data_dict: Dict[int, List[Tuple[str, int]]],
    metadata: Dict[str, Any],
    m_values: List[int],
    num_iterations: int = 1000,
    num_threads: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate predictions using independent models.
    
    Each state is predicted independently without borrowing strength.
    Uses multithreading to accelerate sampling iterations.
    
    Args:
        models: List of fitted independent HPYP models
        fit_data_dict: Training data {group_id: [(name, count), ...]}
        metadata: Metadata dictionary
        m_values: List of prediction sizes to evaluate
        num_iterations: Number of prediction iterations
        num_threads: Number of threads for parallel sampling
        verbose: Whether to show progress
        
    Returns:
        Dictionary with prediction results for each group and m value
    """
    if verbose:
        print("\n" + "="*60)
        print(f"INDEPENDENT PREDICTION (threads={num_threads})")
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
        for m in tqdm(m_values, desc=f"Group {group_id}", disable=not verbose):
            # Prepare tasks for parallel execution
            tasks = [(group_id, model, m, observed_names) for _ in range(num_iterations)]
            
            if num_threads > 1:
                # Parallel execution with progress tracking
                # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
                L_0_values = [None] * num_iterations
                with ProcessPoolExecutor(max_workers=num_threads) as executor:
                    future_to_idx = {executor.submit(_predict_iteration_independent, task): i 
                                    for i, task in enumerate(tasks)}
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        L_0_values[idx] = future.result()
            else:
                # Sequential execution
                L_0_values = [_predict_iteration_independent(task) for task in tasks]
            
            results[group_id][m].extend(L_0_values)
    
    if verbose:
        print("\n✓ Independent predictions complete!")
    
    return results


def predict_dependent(
    model: HierarchicalPitmanYorProcess,
    fit_data_dict: Dict[int, List[Tuple[str, int]]],
    metadata: Dict[str, Any],
    m_values: List[int],
    num_iterations: int = 1000,
    num_threads: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate predictions using dependent model.
    
    All states share the base distribution, enabling borrowing of strength.
    Tracks multiple types of new species:
    - L^(0,0): New to all groups
    - L^(i,j): Seen in group j but new to group i
    - L^0: Total new to group i
    Uses multithreading to accelerate sampling iterations.
    
    Args:
        model: Fitted dependent HPYP model
        fit_data_dict: Training data {group_id: [(name, count), ...]}
        metadata: Metadata dictionary
        m_values: List of prediction sizes to evaluate
        num_iterations: Number of prediction iterations
        num_threads: Number of threads for parallel sampling
        verbose: Whether to show progress
        
    Returns:
        Dictionary with detailed prediction results
    """
    if verbose:
        print("\n" + "="*60)
        print(f"DEPENDENT PREDICTION (threads={num_threads})")
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
        # Prepare tasks for parallel execution
        tasks = [(model, m, num_groups, all_observed_names, observed_names_per_group) 
                 for _ in range(num_iterations)]
        
        if num_threads > 1:
            # Parallel execution with progress tracking
            # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
            # ThreadPoolExecutor doesn't help with CPU-bound tasks
            iteration_results_list = [None] * num_iterations
            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                future_to_idx = {executor.submit(_predict_iteration_dependent, task): i 
                                for i, task in enumerate(tasks)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    iteration_results_list[idx] = future.result()
        else:
            # Sequential execution
            iteration_results_list = [_predict_iteration_dependent(task) for task in tasks]
        
        # Combine results
        for iteration_results in iteration_results_list:
            for group_id in range(num_groups):
                results[group_id][m]['L_0_0'].append(iteration_results[group_id]['L_0_0'])
                results[group_id][m]['L_0'].append(iteration_results[group_id]['L_0'])
                for other in iteration_results[group_id]['L_from_other']:
                    results[group_id][m]['L_from_other'][other].append(
                        iteration_results[group_id]['L_from_other'][other]
                    )
    
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

