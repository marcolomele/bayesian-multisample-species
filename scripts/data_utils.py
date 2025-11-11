"""
Data loading and preparation for HPYP experiments.

CRF terminology:
- Restaurants: Groups/samples
- Dishes: Species/atoms  
- Customers: Individual observations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter


def load_data(
    data_path: str,
    restaurants: Optional[List[str]] = None,
    max_customers: Optional[int] = None,
    restaurant_col: Optional[str] = None,
    dish_col: Optional[str] = None,
    count_col: Optional[str] = None,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Load CSV data and convert to standardized format.
    
    Returns DataFrame with columns: [restaurant, dish, total_count]
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Map column names if provided
    if restaurant_col or dish_col or count_col:
        rename_map = {}
        if restaurant_col: rename_map[restaurant_col] = 'restaurant'
        if dish_col: rename_map[dish_col] = 'dish'
        if count_col: rename_map[count_col] = 'count'
        
        missing = [c for c in rename_map.keys() if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")
        df = df.rename(columns=rename_map)
    
    # Validate required columns
    required = ['restaurant', 'dish', 'count']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                       f"Specify column mapping: restaurant_col, dish_col, count_col")
    
    # Filter by restaurants
    if restaurants is not None:
        df = df[df['restaurant'].isin(restaurants)].copy()
        print(f"Filtered to {restaurants}: {len(df):,} records")
    
    # Cap customers per restaurant
    if max_customers is not None:
        print(f"Capping: max {max_customers:,} per restaurant")
        sampled = []
        for restaurant in df['restaurant'].unique():
            rest_df = df[df['restaurant'] == restaurant].copy()
            total = rest_df['count'].sum()
            
            if total <= max_customers:
                print(f"  {restaurant}: {total:,} (using all)")
                sampled.append(rest_df)
            else:
                print(f"  {restaurant}: sampling {max_customers:,} from {total:,}")
                # Expand to individuals, sample, aggregate
                expanded = []
                for _, row in rest_df.iterrows():
                    expanded.extend([row['dish']] * row['count'])
                
                indices = np.random.choice(len(expanded), max_customers, replace=False)
                sampled_dishes = [expanded[i] for i in indices]
                counts = Counter(sampled_dishes)
                
                sampled.append(pd.DataFrame([
                    {'restaurant': restaurant, 'dish': d, 'count': c}
                    for d, c in counts.items()
                ]))
        
        df = pd.concat(sampled, ignore_index=True)
    
    # Aggregate by (restaurant, dish)
    result = df.groupby(['restaurant', 'dish'], as_index=False)['count'].sum()
    result.rename(columns={'count': 'total_count'}, inplace=True)
    
    print(f"\nLoaded {len(result)} unique (restaurant, dish) pairs")
    print(f"Restaurants: {sorted(result['restaurant'].unique().tolist())}")
    print(f"Total customers: {result['total_count'].sum():,}")
    
    return result


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split customers into fit and test sets.
    
    Returns (fit_data, test_data) DataFrames with [restaurant, dish, count]
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nSplitting: train_ratio={train_ratio}")
    
    # Auto-detect column names
    if 'restaurant' in df.columns and 'dish' in df.columns:
        rest_col, dish_col = 'restaurant', 'dish'
        count_col = 'total_count' if 'total_count' in df.columns else 'count'
    elif 'state' in df.columns and 'name' in df.columns:
        rest_col, dish_col = 'state', 'name'
        count_col = 'total_count' if 'total_count' in df.columns else 'count'
    else:
        rest_col, dish_col, count_col = df.columns[0], df.columns[1], df.columns[2]
    
    fit_rows, test_rows = [], []
    
    for _, row in df.iterrows():
        total = row[count_col]
        # Binomial split
        fit_count = np.random.binomial(total, train_ratio)
        test_count = total - fit_count
        
        if fit_count > 0:
            fit_rows.append({rest_col: row[rest_col], dish_col: row[dish_col], 'count': fit_count})
        if test_count > 0:
            test_rows.append({rest_col: row[rest_col], dish_col: row[dish_col], 'count': test_count})
    
    fit_data = pd.DataFrame(fit_rows)
    test_data = pd.DataFrame(test_rows)
    
    fit_total = fit_data['count'].sum() if len(fit_data) > 0 else 0
    test_total = test_data['count'].sum() if len(test_data) > 0 else 0
    
    print(f"Fit: {len(fit_data)} pairs, {fit_total:,} customers")
    print(f"Test: {len(test_data)} pairs, {test_total:,} customers")
    
    return fit_data, test_data


def prepare_input(
    df: pd.DataFrame,
    restaurants: List[str]
) -> Tuple[Dict[int, List[Tuple[str, int]]], Dict[str, Any]]:
    """
    Convert DataFrame to HPYP input format.
    
    Returns:
        - data_dict: {restaurant_id: [(dish, count), ...]}
        - metadata: {restaurants, mappings, counts, ...}
    """
    # Auto-detect column names
    if 'restaurant' in df.columns and 'dish' in df.columns:
        rest_col, dish_col = 'restaurant', 'dish'
        count_col = 'count'
    elif 'state' in df.columns and 'name' in df.columns:
        rest_col, dish_col = 'state', 'name'
        count_col = 'count' if 'count' in df.columns else 'total_count'
    else:
        rest_col, dish_col = df.columns[0], df.columns[1]
        count_col = df.columns[2] if len(df.columns) > 2 else 'count'
    
    # Create mappings
    restaurant_to_id = {r: i for i, r in enumerate(restaurants)}
    id_to_restaurant = {i: r for i, r in enumerate(restaurants)}
    
    # Organize data
    data_dict = defaultdict(list)
    for _, row in df.iterrows():
        restaurant = row[rest_col]
        if restaurant in restaurant_to_id:
            restaurant_id = restaurant_to_id[restaurant]
            data_dict[restaurant_id].append((row[dish_col], row[count_col]))
    
    data_dict = dict(data_dict)
    
    # Compute metadata
    n_i = {}  # Total customers per restaurant
    K_n_i = {}  # Unique dishes per restaurant
    all_dishes = set()
    dishes_per_restaurant = {}  # Dishes observed in each restaurant
    
    for rid, observations in data_dict.items():
        n_i[rid] = sum(count for _, count in observations)
        K_n_i[rid] = len(observations)
        restaurant_dishes = set(dish for dish, _ in observations)
        dishes_per_restaurant[rid] = restaurant_dishes
        all_dishes.update(restaurant_dishes)
    
    # Calculate shared dishes (intersection across all restaurants)
    if len(dishes_per_restaurant) > 1:
        shared_dishes = set.intersection(*dishes_per_restaurant.values())
    elif len(dishes_per_restaurant) == 1:
        # If only one restaurant, all its dishes are "shared"
        shared_dishes = list(dishes_per_restaurant.values())[0]
    else:
        shared_dishes = set()
    
    metadata = {
        'restaurants': restaurants,
        'restaurant_to_id': restaurant_to_id,
        'id_to_restaurant': id_to_restaurant,
        'n_i': n_i,
        'K_n_i': K_n_i,
        'all_dishes': all_dishes,
        'shared_dishes': shared_dishes,
        'num_restaurants': len(restaurants),
        'num_groups': len(restaurants),
        # Legacy aliases for compatibility
        'group_to_state': id_to_restaurant,
        'state_to_group': restaurant_to_id
    }
    
    print("\nData summary per restaurant:")
    for rid in sorted(data_dict.keys()):
        r = id_to_restaurant[rid]
        print(f"  Restaurant {rid} ({r}): {n_i[rid]:,} customers, {K_n_i[rid]:,} dishes")
    print(f"Total unique dishes: {len(all_dishes):,}")
    print(f"Shared dishes (across all restaurants): {len(shared_dishes):,}")
    
    return data_dict, metadata


def expand_customers(data_dict: Dict[int, List[Tuple[str, int]]]) -> Dict[int, List[str]]:
    """
    Expand (dish, count) format to individual customers.
    
    Returns {restaurant_id: [dish, dish, dish, ...]}
    """
    expanded = {}
    for rid, observations in data_dict.items():
        expanded_list = []
        for dish, count in observations:
            expanded_list.extend([dish] * count)
        expanded[rid] = expanded_list
    return expanded
