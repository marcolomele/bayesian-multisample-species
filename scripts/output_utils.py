"""
Output generation utilities for creating tables and summaries.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
from model_fitting import get_parameter_estimates


def create_independent_table(
    stats: Dict[str, Any],
    m_values: List[int],
    metadata: Dict[str, Any],
    output_path: str = None
) -> pd.DataFrame:
    """
    Create table for independent model predictions (like Table 2 in paper).
    
    Format:
    | m | State1_L0 | State1_HPD | State2_L0 | State2_HPD | ...
    
    Args:
        stats: Summary statistics from compute_statistics
        m_values: List of m values
        metadata: Metadata with state information
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with formatted results
    """
    num_groups = metadata['num_groups']
    
    rows = []
    for m in m_values:
        row = {'m': m}
        
        for group_id in range(num_groups):
            state = metadata['group_to_state'][group_id]
            L_0_mean = stats[group_id][m]['L_0_mean']
            hpd_lower, hpd_upper = stats[group_id][m]['L_0_hpd']
            
            row[f'{state}_L0'] = f"{L_0_mean:.2f}"
            row[f'{state}_HPD'] = f"({int(hpd_lower)}, {int(hpd_upper)})"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Independent predictions table saved to {output_path}")
    
    return df


def create_dependent_table(
    stats: Dict[str, Any],
    m_values: List[int],
    metadata: Dict[str, Any],
    output_path: str = None
) -> pd.DataFrame:
    """
    Create table for dependent model predictions (like Table 4 in paper).
    
    Format:
    | m | State | L^(0,0) | L^(0,j) [for each j] | L^0 | HPD |
    
    Args:
        stats: Summary statistics from compute_statistics (dependent)
        m_values: List of m values
        metadata: Metadata with state information
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with formatted results
    """
    num_groups = metadata['num_groups']
    
    rows = []
    for m in m_values:
        for group_id in range(num_groups):
            state = metadata['group_to_state'][group_id]
            
            row = {
                'm': m,
                'state': state,
                'L_0_0': f"{stats[group_id][m]['L_0_0_mean']:.2f}",
            }
            
            # Add L^(0,j) for each other group j
            for other_group in range(num_groups):
                if other_group != group_id:
                    other_state = metadata['group_to_state'][other_group]
                    if other_group in stats[group_id][m]['L_from_other']:
                        value = stats[group_id][m]['L_from_other'][other_group]['mean']
                        row[f'L_from_{other_state}'] = f"{value:.2f}"
            
            # Total new to this group
            L_0_mean = stats[group_id][m]['L_0_mean']
            hpd_lower, hpd_upper = stats[group_id][m]['L_0_hpd']
            
            row['L_0'] = f"{L_0_mean:.2f}"
            row['HPD'] = f"({int(hpd_lower)}, {int(hpd_upper)})"
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Dependent predictions table saved to {output_path}")
    
    return df


def create_parameter_table(
    independent_models: List[Any],
    dependent_model: Any,
    metadata: Dict[str, Any],
    output_path: str = None
) -> pd.DataFrame:
    """
    Create table comparing parameter estimates.
    
    Args:
        independent_models: List of fitted independent models
        dependent_model: Fitted dependent model
        metadata: Metadata with state information
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with parameter estimates
    """
    rows = []
    
    # Independent model parameters
    for i, model in enumerate(independent_models):
        state = metadata['group_to_state'][i]
        estimates = get_parameter_estimates(model)
        
        rows.append({
            'model': 'Independent',
            'state': state,
            'theta_0': f"{estimates['theta_0_mean']:.2f} ± {estimates['theta_0_std']:.2f}",
            'theta_j': f"{estimates['theta_j_mean']:.2f} ± {estimates['theta_j_std']:.2f}",
            'd_0': f"{estimates['d_0_mean']:.4f} ± {estimates['d_0_std']:.4f}",
            'd_j': f"{estimates['d_j_mean']:.4f} ± {estimates['d_j_std']:.4f}",
            'num_base_tables': f"{estimates['num_base_tables_mean']:.1f}",
            'num_unique_dishes': f"{estimates['num_unique_dishes_mean']:.1f}"
        })
    
    # Dependent model parameters (shared across all states)
    dep_estimates = get_parameter_estimates(dependent_model)
    rows.append({
        'model': 'Dependent',
        'state': 'ALL (shared)',
        'theta_0': f"{dep_estimates['theta_0_mean']:.2f} ± {dep_estimates['theta_0_std']:.2f}",
        'theta_j': f"{dep_estimates['theta_j_mean']:.2f} ± {dep_estimates['theta_j_std']:.2f}",
        'd_0': f"{dep_estimates['d_0_mean']:.4f} ± {dep_estimates['d_0_std']:.4f}",
        'd_j': f"{dep_estimates['d_j_mean']:.4f} ± {dep_estimates['d_j_std']:.4f}",
        'num_base_tables': f"{dep_estimates['num_base_tables_mean']:.1f}",
        'num_unique_dishes': f"{dep_estimates['num_unique_dishes_mean']:.1f}"
    })
    
    df = pd.DataFrame(rows)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Parameter estimates table saved to {output_path}")
    
    return df


def create_comparison_table(
    stats_independent: Dict[str, Any],
    stats_dependent: Dict[str, Any],
    m_values: List[int],
    metadata: Dict[str, Any],
    output_path: str = None
) -> pd.DataFrame:
    """
    Create comparison table showing borrowing of strength effect.
    
    Compares HPD interval widths and predictions between models.
    
    Args:
        stats_independent: Independent model statistics
        stats_dependent: Dependent model statistics
        m_values: List of m values
        metadata: Metadata with state information
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    
    for m in m_values:
        for group_id in range(metadata['num_groups']):
            state = metadata['group_to_state'][group_id]
            
            # Independent model
            ind_mean = stats_independent[group_id][m]['L_0_mean']
            ind_hpd_lower, ind_hpd_upper = stats_independent[group_id][m]['L_0_hpd']
            ind_hpd_width = ind_hpd_upper - ind_hpd_lower
            
            # Dependent model
            dep_mean = stats_dependent[group_id][m]['L_0_mean']
            dep_hpd_lower, dep_hpd_upper = stats_dependent[group_id][m]['L_0_hpd']
            dep_hpd_width = dep_hpd_upper - dep_hpd_lower
            
            # Relative reduction in uncertainty
            width_reduction = ((ind_hpd_width - dep_hpd_width) / ind_hpd_width) * 100
            
            rows.append({
                'm': m,
                'state': state,
                'independent_mean': f"{ind_mean:.2f}",
                'dependent_mean': f"{dep_mean:.2f}",
                'mean_difference': f"{abs(ind_mean - dep_mean):.2f}",
                'independent_hpd_width': f"{ind_hpd_width:.1f}",
                'dependent_hpd_width': f"{dep_hpd_width:.1f}",
                'width_reduction_pct': f"{width_reduction:.1f}%"
            })
    
    df = pd.DataFrame(rows)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Comparison table saved to {output_path}")
    
    return df


def save_linearity_check(
    linearity_results: Dict[int, Dict[str, float]],
    metadata: Dict[str, Any],
    output_path: str
):
    """
    Save linearity check results to text file.
    
    Args:
        linearity_results: Results from check_linearity
        metadata: Metadata with state information
        output_path: Path to save text file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LINEARITY CHECK (Theorem 1 Verification)\n")
        f.write("="*60 + "\n\n")
        f.write("Testing if L_m is linear in m:\n")
        f.write("  L_m ≈ slope * m + intercept\n\n")
        
        for group_id, results in linearity_results.items():
            state = metadata['group_to_state'][group_id]
            f.write(f"\nGroup {group_id} ({state}):\n")
            f.write(f"  Slope:       {results['slope']:.4f}\n")
            f.write(f"  Intercept:   {results['intercept']:.4f}\n")
            f.write(f"  R²:          {results['r_squared']:.6f}\n")
            f.write(f"  p-value:     {results['p_value']:.2e}\n")
            f.write(f"  Std Error:   {results['std_err']:.4f}\n")
            
            if results['r_squared'] > 0.99:
                f.write("  ✓ Excellent linear fit (R² > 0.99)\n")
            elif results['r_squared'] > 0.95:
                f.write("  ✓ Good linear fit (R² > 0.95)\n")
            else:
                f.write("  ⚠ Moderate linear fit\n")
    
    print(f"✓ Linearity check saved to {output_path}")


def print_summary(
    stats_independent: Dict[str, Any],
    stats_dependent: Dict[str, Any],
    metadata: Dict[str, Any],
    m_value: int = 1000
):
    """
    Print a summary of results to console.
    
    Args:
        stats_independent: Independent model statistics
        stats_dependent: Dependent model statistics
        metadata: Metadata
        m_value: Which m value to summarize (default 1000)
    """
    print("\n" + "="*60)
    print(f"RESULTS SUMMARY (m = {m_value})")
    print("="*60)
    
    print("\nIndependent Model:")
    print("-" * 40)
    for group_id in range(metadata['num_groups']):
        state = metadata['group_to_state'][group_id]
        mean = stats_independent[group_id][m_value]['L_0_mean']
        hpd_lower, hpd_upper = stats_independent[group_id][m_value]['L_0_hpd']
        print(f"  {state}: L̂^0 = {mean:.2f}, HPD = ({int(hpd_lower)}, {int(hpd_upper)})")
    
    print("\nDependent Model (Borrowing Strength):")
    print("-" * 40)
    for group_id in range(metadata['num_groups']):
        state = metadata['group_to_state'][group_id]
        mean = stats_dependent[group_id][m_value]['L_0_mean']
        hpd_lower, hpd_upper = stats_dependent[group_id][m_value]['L_0_hpd']
        print(f"  {state}: L̂^0 = {mean:.2f}, HPD = ({int(hpd_lower)}, {int(hpd_upper)})")
    
    print("\nBorrowing Strength Effect:")
    print("-" * 40)
    for group_id in range(metadata['num_groups']):
        state = metadata['group_to_state'][group_id]
        ind_hpd = stats_independent[group_id][m_value]['L_0_hpd']
        dep_hpd = stats_dependent[group_id][m_value]['L_0_hpd']
        ind_width = ind_hpd[1] - ind_hpd[0]
        dep_width = dep_hpd[1] - dep_hpd[0]
        reduction = ((ind_width - dep_width) / ind_width) * 100
        print(f"  {state}: HPD width reduction = {reduction:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    print("Output utilities module loaded successfully!")
    print("Run experiment.py to test the complete pipeline.")

