"""
Data loading and preparation utilities for HPYP experiments.

Uses Chinese Restaurant Franchise (CRF) terminology:
- Restaurants: Groups/samples (e.g., states, populations)
- Dishes: Species/atoms (e.g., names, genes, words)
- Customers: Individual observations
- Tables: Clusters within a restaurant serving the same dish
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict


def load_species_data(
    data: Union[str, pd.DataFrame],
    restaurant_col: str = 'restaurant',
    dish_col: str = 'dish',
    count_col: str = 'count',
    max_customers_per_restaurant: Optional[int] = None,
    restaurants: Optional[List[str]] = None,
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Load species sampling data in CRF format.
    
    Generic loader that works with any species sampling data where:
    - Restaurants (groups) contain dishes (species) with customer counts
    
    Args:
        data: Path to CSV file or DataFrame with columns [restaurant, dish, count]
        restaurant_col: Column name for restaurants (default: 'restaurant')
        dish_col: Column name for dishes (default: 'dish')
        count_col: Column name for customer counts (default: 'count')
        max_customers_per_restaurant: Optional cap on customers per restaurant
        restaurants: Optional list of restaurants to include
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns [restaurant, dish, total_count]
        Ready for HPYP input preparation
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Load data
    if isinstance(data, str):
        print(f"Loading data from {data}...")
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    # Rename columns to standard names
    df = df.rename(columns={
        restaurant_col: 'restaurant',
        dish_col: 'dish',
        count_col: 'count'
    })
    
    # Filter by restaurants if specified
    if restaurants is not None:
        df = df[df['restaurant'].isin(restaurants)].copy()
        print(f"Filtered to restaurants {restaurants}: {len(df):,} records")
    
    # Cap customers per restaurant if specified
    if max_customers_per_restaurant is not None:
        print(f"Capping customers: max {max_customers_per_restaurant:,} per restaurant")
        
        sampled_rows = []
        for restaurant in df['restaurant'].unique():
            restaurant_df = df[df['restaurant'] == restaurant].copy()
            total_customers = restaurant_df['count'].sum()
            
            if total_customers <= max_customers_per_restaurant:
                print(f"  {restaurant}: {total_customers:,} customers (using all)")
                sampled_df = restaurant_df
            else:
                # Sample proportionally from each dish
                print(f"  {restaurant}: {total_customers:,} total, sampling {max_customers_per_restaurant:,}")
                
                # Expand to individual customers
                expanded = []
                for _, row in restaurant_df.iterrows():
                    expanded.extend([row['dish']] * row['count'])
                
                # Random sample without replacement
                if len(expanded) > max_customers_per_restaurant:
                    sampled_indices = np.random.choice(
                        len(expanded), 
                        size=max_customers_per_restaurant, 
                        replace=False
                    )
                    sampled_dishes = [expanded[i] for i in sampled_indices]
                else:
                    sampled_dishes = expanded
                
                # Aggregate back
                from collections import Counter
                dish_counts = Counter(sampled_dishes)
                
                sampled_df = pd.DataFrame([
                    {'restaurant': restaurant, 'dish': dish, 'count': count}
                    for dish, count in dish_counts.items()
                ])
                
                actual_total = sampled_df['count'].sum()
                print(f"    Sampled: {actual_total:,} customers")
            
            sampled_rows.append(sampled_df)
        
        df = pd.concat(sampled_rows, ignore_index=True)
    
    # Aggregate by (restaurant, dish)
    result = df.groupby(['restaurant', 'dish'], as_index=False)['count'].sum()
    result.rename(columns={'count': 'total_count'}, inplace=True)
    
    print(f"\nLoaded {len(result)} unique (restaurant, dish) pairs")
    print(f"Restaurants: {sorted(result['restaurant'].unique().tolist())}")
    print(f"Total customers: {result['total_count'].sum():,}")
    
    return result


def load_names_data_stratified(
    filepath: str,
    year: Optional[int] = None,
    states: List[str] = None,
    max_observations_per_state: int = 100000,
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Load names data with stratified sampling by sex, capped per state.
    
    Maintains sex proportions while limiting total observations per state.
    
    Args:
        filepath: Path to namesbystate_subset.csv
        year: Optional year to filter by
        states: List of states to include (default: all)
        max_observations_per_state: Maximum observations per state (default: 100k)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns [state, name, total_count]
        Aggregated across years and sexes, with stratified sampling applied
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Filter by year if specified
    if year is not None:
        df = df[df['year'] == year].copy()
        print(f"Filtered to year {year}: {len(df):,} records")
    
    # Filter by states if specified
    if states is not None:
        df = df[df['state'].isin(states)].copy()
        print(f"Filtered to states {states}: {len(df):,} records")
    
    print(f"Stratified sampling: max {max_observations_per_state:,} per state")
    
    # Process each state separately
    sampled_rows = []
    
    for state in (states if states else df['state'].unique()):
        state_df = df[df['state'] == state].copy()
        
        if len(state_df) == 0:
            continue
        
        total_state_obs = state_df['count'].sum()
        
        if total_state_obs <= max_observations_per_state:
            # No need to sample, use all data
            print(f"  {state}: {total_state_obs:,} observations (using all)")
            # Aggregate by (state, name) across sex
            sampled_df = state_df.groupby(['state', 'name'], as_index=False)['count'].sum()
        else:
            # Need to sample, maintain sex proportions
            # Calculate sex proportions
            sex_totals = state_df.groupby('sex')['count'].sum()
            sex_proportions = sex_totals / sex_totals.sum()
            
            print(f"  {state}: {total_state_obs:,} total, sampling {max_observations_per_state:,}")
            print(f"    Sex proportions: {dict(sex_proportions)}")
            
            # Sample proportionally from each sex
            sampled_rows_state = []
            for sex, proportion in sex_proportions.items():
                sex_df = state_df[state_df['sex'] == sex].copy()
                sex_target = int(max_observations_per_state * proportion)
                sex_total = sex_df['count'].sum()
                
                if sex_total <= sex_target:
                    # Use all data for this sex
                    for _, row in sex_df.iterrows():
                        sampled_rows_state.append({
                            'state': state,
                            'name': row['name'],
                            'count': row['count']
                        })
                else:
                    # Sample proportionally from each (name, count) pair
                    # Expand to individual observations
                    expanded = []
                    for _, row in sex_df.iterrows():
                        expanded.extend([row['name']] * row['count'])
                    
                    # Random sample without replacement
                    if len(expanded) > sex_target:
                        sampled_indices = np.random.choice(len(expanded), size=sex_target, replace=False)
                        sampled_names = [expanded[i] for i in sampled_indices]
                    else:
                        sampled_names = expanded
                    
                    # Aggregate back
                    from collections import Counter
                    name_counts = Counter(sampled_names)
                    
                    for name, count in name_counts.items():
                        sampled_rows_state.append({
                            'state': state,
                            'name': name,
                            'count': count
                        })
            
            sampled_df = pd.DataFrame(sampled_rows_state)
            if len(sampled_df) > 0:
                sampled_df = sampled_df.groupby(['state', 'name'], as_index=False)['count'].sum()
            
            actual_total = sampled_df['count'].sum() if len(sampled_df) > 0 else 0
            print(f"    Sampled: {actual_total:,} observations")
        
        sampled_rows.append(sampled_df)
    
    # Combine all states
    if sampled_rows:
        result = pd.concat(sampled_rows, ignore_index=True)
        result = result.groupby(['state', 'name'], as_index=False)['count'].sum()
        result.rename(columns={'count': 'total_count'}, inplace=True)
    else:
        result = pd.DataFrame(columns=['state', 'name', 'total_count'])
    
    year_str = f" (year {year})" if year else ""
    print(f"\nLoaded {len(result)} unique (state, name) pairs{year_str}")
    print(f"States: {sorted(result['state'].unique().tolist())}")
    print(f"Total observations: {result['total_count'].sum():,}")
    
    return result


def load_names_data(filepath: str, year: Optional[int] = None) -> pd.DataFrame:
    """
    Load and aggregate names data from CSV.
    
    Args:
        filepath: Path to namesbystate_subset.csv
        year: Optional year to filter by (e.g., 2024). If None, uses all years.
        
    Returns:
        DataFrame with columns [state, name, total_count]
        Aggregated across years and sexes (or just for specified year)
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Filter by year if specified
    if year is not None:
        df = df[df['year'] == year].copy()
        print(f"Filtered to year {year}: {len(df):,} records")
    
    # Aggregate counts by (state, name) across years and sex
    aggregated = df.groupby(['state', 'name'], as_index=False)['count'].sum()
    aggregated.rename(columns={'count': 'total_count'}, inplace=True)
    
    year_str = f" (year {year})" if year else ""
    print(f"Loaded {len(aggregated)} unique (state, name) pairs from {len(df)} records{year_str}")
    print(f"States: {sorted(aggregated['state'].unique().tolist())}")
    print(f"Total observations: {aggregated['total_count'].sum():,}")
    
    return aggregated


def split_customers_random(
    df: pd.DataFrame,
    restaurant_col: str = 'restaurant',
    dish_col: str = 'dish',
    count_col: str = 'total_count',
    train_ratio: float = 0.8,
    random_seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split customers into fit (train) and predict (test) sets.
    
    For each (restaurant, dish) pair with N customers, we randomly assign
    approximately train_ratio*N customers to fit and the rest to predict.
    
    Args:
        df: DataFrame with columns [restaurant, dish, total_count] (or custom names)
        restaurant_col: Column name for restaurants (default: 'restaurant')
        dish_col: Column name for dishes (default: 'dish')
        count_col: Column name for customer counts (default: 'total_count')
        train_ratio: Proportion of customers for training (default 0.8)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (fit_data, predict_data) DataFrames
        Each with columns [restaurant, dish, count]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"Splitting customers with train_ratio={train_ratio}, seed={random_seed}")
    
    fit_rows = []
    predict_rows = []
    
    for _, row in df.iterrows():
        restaurant = row[restaurant_col]
        dish = row[dish_col]
        total_customers = row[count_col]
        
        # Use binomial distribution to split customers
        # Each customer has probability train_ratio of being in training set
        fit_customers = np.random.binomial(total_customers, train_ratio)
        predict_customers = total_customers - fit_customers
        
        if fit_customers > 0:
            fit_rows.append({
                restaurant_col: restaurant,
                dish_col: dish,
                'count': fit_customers
            })
        
        if predict_customers > 0:
            predict_rows.append({
                restaurant_col: restaurant,
                dish_col: dish,
                'count': predict_customers
            })
    
    fit_data = pd.DataFrame(fit_rows)
    predict_data = pd.DataFrame(predict_rows)
    
    total_fit = fit_data['count'].sum() if len(fit_data) > 0 else 0
    total_predict = predict_data['count'].sum() if len(predict_data) > 0 else 0
    
    print(f"Fit data: {len(fit_data)} (restaurant, dish) pairs, {total_fit:,} customers")
    print(f"Predict data: {len(predict_data)} (restaurant, dish) pairs, {total_predict:,} customers")
    
    return fit_data, predict_data


def split_data_random(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy function: Randomly split data into fit and predict sets.
    
    Automatically detects column names (supports both CRF and legacy formats).
    """
    # Auto-detect column names
    if 'restaurant' in df.columns and 'dish' in df.columns:
        restaurant_col = 'restaurant'
        dish_col = 'dish'
        count_col = 'total_count' if 'total_count' in df.columns else 'count'
    elif 'state' in df.columns and 'name' in df.columns:
        restaurant_col = 'state'
        dish_col = 'name'
        count_col = 'total_count' if 'total_count' in df.columns else 'count'
    else:
        # Try first three columns
        cols = df.columns.tolist()
        restaurant_col = cols[0]
        dish_col = cols[1]
        count_col = cols[2] if len(cols) > 2 else 'count'
    
    return split_customers_random(
        df, restaurant_col, dish_col, count_col, train_ratio, random_seed
    )


def prepare_crf_input(
    df: pd.DataFrame,
    restaurants: List[str],
    restaurant_col: str = 'restaurant',
    dish_col: str = 'dish',
    count_col: str = 'count'
) -> Tuple[Dict[int, List[Tuple[str, int]]], Dict[str, Any]]:
    """
    Convert DataFrame to HPYP input format using CRF terminology.
    
    Args:
        df: DataFrame with columns [restaurant, dish, count] (or custom names)
        restaurants: List of restaurant IDs in desired order
        restaurant_col: Column name for restaurants (default: 'restaurant')
        dish_col: Column name for dishes (default: 'dish')
        count_col: Column name for customer counts (default: 'count')
        
    Returns:
        Tuple of:
        - data_dict: {restaurant_id: [(dish, customer_count), ...]}
        - metadata: {
            'restaurants': list of restaurant IDs,
            'restaurant_to_id': {restaurant: restaurant_id},
            'id_to_restaurant': {restaurant_id: restaurant},
            'n_i': {restaurant_id: total customers},
            'K_n_i': {restaurant_id: unique dishes},
            'all_dishes': set of all unique dishes,
            'num_restaurants': number of restaurants
          }
    """
    # Create mappings
    restaurant_to_id = {restaurant: i for i, restaurant in enumerate(restaurants)}
    id_to_restaurant = {i: restaurant for i, restaurant in enumerate(restaurants)}
    
    # Organize data by restaurant
    data_dict = defaultdict(list)
    
    for _, row in df.iterrows():
        restaurant = row[restaurant_col]
        dish = row[dish_col]
        customers = row[count_col]
        
        if restaurant in restaurant_to_id:
            restaurant_id = restaurant_to_id[restaurant]
            data_dict[restaurant_id].append((dish, customers))
    
    # Convert to regular dict
    data_dict = dict(data_dict)
    
    # Compute metadata
    n_i = {}  # Total customers per restaurant
    K_n_i = {}  # Unique dishes per restaurant
    all_dishes = set()
    
    for restaurant_id, observations in data_dict.items():
        n_i[restaurant_id] = sum(count for _, count in observations)
        K_n_i[restaurant_id] = len(observations)
        all_dishes.update(dish for dish, _ in observations)
    
    metadata = {
        'restaurants': restaurants,
        'restaurant_to_id': restaurant_to_id,
        'id_to_restaurant': id_to_restaurant,
        'n_i': n_i,
        'K_n_i': K_n_i,
        'all_dishes': all_dishes,
        'num_restaurants': len(restaurants),
        'num_groups': len(restaurants)  # Alias for compatibility
    }
    
    print("\nData summary per restaurant (CRF):")
    for restaurant_id in sorted(data_dict.keys()):
        restaurant = id_to_restaurant[restaurant_id]
        print(f"  Restaurant {restaurant_id} ({restaurant}): {n_i[restaurant_id]:,} customers, {K_n_i[restaurant_id]:,} unique dishes")
    print(f"Total unique dishes across all restaurants: {len(all_dishes):,}")
    
    return data_dict, metadata


def prepare_hpyp_input(
    df: pd.DataFrame,
    states: List[str]
) -> Tuple[Dict[int, List[Tuple[str, int]]], Dict[str, Any]]:
    """
    Legacy function: Convert DataFrame to HPYP input format.
    
    Automatically detects column names and uses CRF terminology internally.
    """
    # Auto-detect column names
    if 'restaurant' in df.columns and 'dish' in df.columns:
        restaurant_col = 'restaurant'
        dish_col = 'dish'
        count_col = 'count'
        restaurants = states  # Use states as restaurant IDs
    elif 'state' in df.columns and 'name' in df.columns:
        restaurant_col = 'state'
        dish_col = 'name'
        count_col = 'count' if 'count' in df.columns else 'total_count'
        restaurants = states
    else:
        # Try first three columns
        cols = df.columns.tolist()
        restaurant_col = cols[0]
        dish_col = cols[1]
        count_col = cols[2] if len(cols) > 2 else 'count'
        restaurants = states
    
    data_dict, metadata = prepare_crf_input(
        df, restaurants, restaurant_col, dish_col, count_col
    )
    
    # Add legacy aliases for backward compatibility
    if 'state' in df.columns:
        metadata['states'] = metadata['restaurants']
        metadata['group_to_state'] = metadata['id_to_restaurant']
        metadata['state_to_group'] = metadata['restaurant_to_id']
        metadata['all_names'] = metadata['all_dishes']
    
    return data_dict, metadata


def expand_customers(data_dict: Dict[int, List[Tuple[str, int]]]) -> Dict[int, List[str]]:
    """
    Expand compressed (dish, customer_count) format into individual customers.
    
    Args:
        data_dict: {restaurant_id: [(dish, customer_count), ...]}
        
    Returns:
        {restaurant_id: [dish, dish, dish, ...]} with customer counts expanded
    """
    expanded = {}
    
    for restaurant_id, observations in data_dict.items():
        expanded_list = []
        for dish, count in observations:
            expanded_list.extend([dish] * count)
        expanded[restaurant_id] = expanded_list
    
    return expanded


def expand_observations(data_dict: Dict[int, List[Tuple[str, int]]]) -> Dict[int, List[str]]:
    """
    Legacy function: Expand compressed format into individual observations.
    
    Alias for expand_customers() using CRF terminology.
    """
    return expand_customers(data_dict)


def convert_names_to_crf(
    df: pd.DataFrame,
    state_col: str = 'state',
    name_col: str = 'name',
    count_col: str = 'total_count'
) -> pd.DataFrame:
    """
    Convert names data format to CRF terminology.
    
    Args:
        df: DataFrame with [state, name, total_count] columns
        state_col: Column name for states (default: 'state')
        name_col: Column name for names (default: 'name')
        count_col: Column name for counts (default: 'total_count')
        
    Returns:
        DataFrame with [restaurant, dish, total_count] columns
    """
    result = df.copy()
    result = result.rename(columns={
        state_col: 'restaurant',
        name_col: 'dish',
        count_col: 'total_count'
    })
    return result


if __name__ == "__main__":
    # Test the data loading pipeline
    import os
    
    data_path = os.path.join('..', 'data', 'namesbystate_subset.csv')
    
    # Load data
    df = load_names_data(data_path)
    
    # Split data
    fit_data, predict_data = split_data_random(df, train_ratio=0.8, random_seed=42)
    
    # Prepare HPYP input
    states = ['CA', 'FL', 'NY', 'PA', 'TX']
    data_dict, metadata = prepare_hpyp_input(fit_data, states)
    
    print("\n✓ Data loading pipeline test successful!")

