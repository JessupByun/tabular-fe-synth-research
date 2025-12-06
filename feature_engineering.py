"""
Feature engineering pipeline for tabular data.

- Processes a single CSV file at a time.
- Performs feature selection (MI / tree/XGBoost importance / random).
- Applies deterministic transforms (squares, pairwise products, optional log1p).
- Writes augmented CSVs. Engineered columns can be identified by name patterns (_sq, _x_, _log1p).

Design:
- Core method: MI with top-k numeric features (k=5)
- Ablations: tree/XGBoost importance and random top-k
- Transforms: squares + pairwise products on top-k numeric features (pairwise sums excluded)
- Ablation: Full FE + log1p

Note:
- Target column is always the rightmost column in the DataFrame.
- Task type must be manually specified (classification or regression).
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input path
INPUT_CSV = "data/real_data/abalone/train/abalone_train.csv"
TASK_TYPE = "regression" # Options: "classification" | "regression"

# ----------------------------------------------------------------------------
# Feature Selection Settings

#   - "mi":    Select features by mutual information with target (core method)
#   - "tree":  Select features by tree/XGBoost importance (ablation)
#   - "random": Randomly select top-k features (ablation)
FEATURE_SELECTION_METHOD = "tree"

# Number of top features to select
#   - int:  Select top K features (default is 5)
#   - None: Select all numeric features
TOP_K_NUMERIC_FEATURES = 5

# Random seed for reproducibility
#   - Used for random feature selection, tree-based models, and sklearn's MI calculations
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Feature Transform Settings
# Always applies Full FE: squares + pairwise products 
# Note: Pairwise sums are excluded per design

# Apply log1p transform to positive values (ablation)
#   - True:  Adds x_log1p column for each feature (only for positive values)
#   - False: Skip log1p transform
USE_LOG1P = False

# =========================
# IMPORTS
# =========================

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# =========================
# CORE HELPERS: FEATURE SELECTION
# =========================

def select_numeric_features(df: pd.DataFrame, target_col: str) -> List[str]:
    """Pick numeric columns except the target."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c != target_col]

def rank_features_by_mi(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    task_type: str,
) -> pd.Series:
    """
    Rank numeric features by mutual information with the target.
    
    Args:
        df: DataFrame containing features and target
        numeric_cols: List of numeric column names to rank
        target_col: Name of the target column
        task_type: Task type ("classification" | "regression") - must be specified
    
    Returns:
        Series of MI scores sorted in descending order
    """
    if not numeric_cols:
        raise ValueError("No numeric features found to rank")
    
    if task_type not in ("classification", "regression"):
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")

    y = df[target_col]
    X = df[numeric_cols]

    if task_type == "classification":
        mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    else:
        mi_scores = mutual_info_regression(X, y, discrete_features=False, random_state=0)

    mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)
    return mi_series


def rank_features_by_tree_importance(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    task_type: str,
    use_xgboost: bool = True,
) -> pd.Series:
    """
    Rank numeric features by tree-based importance (XGBoost or RandomForest).
    
    Args:
        df: DataFrame containing features and target
        numeric_cols: List of numeric column names to rank
        target_col: Name of the target column
        task_type: Task type ("classification" | "regression") - must be specified
        use_xgboost: If True and XGBoost is available, use XGBoost; otherwise use RandomForest
    
    Returns:
        Series of importance scores sorted in descending order
    """
    if not numeric_cols:
        raise ValueError("No numeric features found to rank")
    
    if task_type not in ("classification", "regression"):
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")

    y = df[target_col]
    X = df[numeric_cols]

    # Use XGBoost if available and requested, otherwise use RandomForest
    if use_xgboost and HAS_XGBOOST:
        if task_type == "classification":
            model = xgb.XGBClassifier(random_state=RANDOM_SEED, n_estimators=100, verbosity=0)
        else:
            model = xgb.XGBRegressor(random_state=RANDOM_SEED, n_estimators=100, verbosity=0)
    else:
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)

    model.fit(X, y)
    importance_scores = model.feature_importances_
    importance_series = pd.Series(importance_scores, index=numeric_cols).sort_values(ascending=False)
    return importance_series

def select_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    method: str,
    top_k: int | None,
    rng: np.random.Generator,
    task_type: str,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select features according to the chosen method.

    Args:
        df: DataFrame containing features and target
        numeric_cols: List of numeric column names
        target_col: Name of the target column
        method: Feature selection method
            - "mi"    : mutual information with target (core method)
            - "tree"  : tree/XGBoost importance (ablation)
            - "random": random subset of numeric_cols (ablation)
        top_k: Number of top features to select (None for all)
        rng: Random number generator
        task_type: Task type ("classification" | "regression") - must be specified.
                   Only used when method in ("mi", "tree").

    Returns:
        selected_cols, scores_dict
    """
    method = method.lower()
    scores: pd.Series

    if method == "mi":
        scores = rank_features_by_mi(df, numeric_cols, target_col, task_type=task_type)
    elif method == "tree":
        scores = rank_features_by_tree_importance(df, numeric_cols, target_col, task_type=task_type)
    elif method == "random":
        if not numeric_cols:
            raise ValueError("No numeric features found for random selection")
        # Assign dummy scores (all equal) for completeness
        scores = pd.Series(1.0, index=numeric_cols)
    else:
        raise ValueError(f"Unknown feature selection method: {method}. Must be 'mi', 'tree', or 'random'.")

    if method == "random":
        if top_k is None or top_k >= len(numeric_cols):
            selected_cols = numeric_cols.copy()
        else:
            selected_cols = list(rng.choice(numeric_cols, size=top_k, replace=False))
    else:
        if top_k is not None and top_k > 0:
            selected_cols = scores.index[:top_k].tolist()
        else:
            selected_cols = scores.index.tolist()

    return selected_cols, scores.to_dict()


# =========================
# CORE HELPERS: TRANSFORMS
# =========================

# Always use Full FE (tier 1): squares + pairwise products
# No tier logic needed - always applies these transforms


def apply_deterministic_transforms(
    df: pd.DataFrame,
    cols: List[str],
    use_squares: bool,
    use_pairwise_products: bool,
    use_log1p: bool,
    log_eps: float = 1e-6,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add engineered columns to df based on selected numeric columns.
    
    Transforms applied:
    - Squares: x_sq for each feature x
    - Pairwise products: x_y for each pair (x, y)
    - Log1p (ablation): x_log1p for positive values only
    
    Note: Pairwise sums are excluded per design.

    Returns:
        df_augmented, engineered_column_names
    """
    df = df.copy()
    engineered_cols: List[str] = []

    # Per-feature transforms
    for c in cols:
        if use_squares:
            new_col = f"{c}_sq"
            df[new_col] = df[c] ** 2
            engineered_cols.append(new_col)

        if use_log1p:
            # Only apply log1p to positive values; others stay NaN
            new_col = f"{c}_log1p"
            positive_mask = df[c] > 0
            df[new_col] = pd.NA
            # Use numpy for log1p
            df.loc[positive_mask, new_col] = np.log1p(df.loc[positive_mask, c] + log_eps)
            engineered_cols.append(new_col)

    # Pairwise products (sums excluded per design)
    if use_pairwise_products:
        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = cols[i], cols[j]
                new_col = f"{c1}_x_{c2}"
                df[new_col] = df[c1] * df[c2]
                engineered_cols.append(new_col)

    return df, engineered_cols


# =========================
# PIPELINE FOR A SINGLE DATAFRAME
# =========================

def run_feature_engineering_on_df(
    df: pd.DataFrame,
    selection_method: str,
    top_k: int | None,
    use_log1p: bool,
    rng: np.random.Generator,
    task_type: str,
) -> pd.DataFrame:
    """
    Run the full FE pipeline on a single DataFrame.
    Always applies Full FE: squares + pairwise products.

    Args:
        df: DataFrame to process
        selection_method: Feature selection method ("mi", "tree", or "random")
        top_k: Number of top features to select (None for all)
        use_log1p: Whether to apply log1p transform (ablation)
        rng: Random number generator
        task_type: Task type ("classification" | "regression") - must be specified

    Returns:
        df_augmented: DataFrame with original and engineered columns
    """
    # Target column is always the rightmost column
    target_col = df.columns[-1]
    numeric_cols = select_numeric_features(df, target_col)

    if task_type not in ("classification", "regression"):
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")

    print(f"Using target column: {target_col} (rightmost column)")
    print(f"Task type: {task_type}")
    print(f"Found {len(numeric_cols)} numeric feature(s): {numeric_cols}")

    selected_cols, scores_dict = select_features(
        df=df,
        numeric_cols=numeric_cols,
        target_col=target_col,
        method=selection_method,
        top_k=top_k,
        rng=rng,
        task_type=task_type,
    )

    print(f"Selected {len(selected_cols)} feature(s) for engineering using '{selection_method}': {selected_cols}")

    # Always apply Full FE: squares + pairwise products
    df_aug, engineered_cols = apply_deterministic_transforms(
        df=df,
        cols=selected_cols,
        use_squares=True,
        use_pairwise_products=True,
        use_log1p=use_log1p,
    )

    print(f"Added {len(engineered_cols)} engineered column(s).")

    return df_aug

# =========================
# OUTPUT PATH GENERATION
# =========================

def generate_output_path(
    input_csv: str | Path,
    selection_method: str,
    top_k: int | None,
    random_seed: int,
    use_log1p: bool,
) -> Path:
    """
    Auto-generate output CSV path based on input path and configuration.
    
    Format: data/FE_train_data/{dataset_name}/FE_{dataset_name}_train_{method}_k{top_k}_seed{seed}[_log1p].csv
    
    Args:
        input_csv: Input CSV file path
        selection_method: Feature selection method
        top_k: Number of top features selected
        random_seed: Random seed used
        use_log1p: Whether log1p transform was applied
    
    Returns:
        Path object for output CSV
    """
    input_path = Path(input_csv)
    
    # Extract dataset name from input path
    # Try to find dataset name in path (e.g., data/real_data/{dataset_name}/train/...)
    parts = input_path.parts
    dataset_name = None
    
    # Look for dataset name in common path structures
    if "real_data" in parts:
        idx = parts.index("real_data")
        if idx + 1 < len(parts):
            dataset_name = parts[idx + 1]
    elif len(parts) >= 2:
        # Fallback: use parent directory name
        dataset_name = input_path.parent.name
    
    if dataset_name is None:
        # Last resort: use input filename without extension
        dataset_name = input_path.stem.replace("_train", "").replace("train", "")
    
    # Build filename components
    filename_parts = [
        "FE",
        dataset_name,
        "train",
        selection_method,
        f"k{top_k if top_k is not None else 'all'}",
        f"seed{random_seed}",
    ]
    
    if use_log1p:
        filename_parts.append("log1p")
    
    filename = "_".join(filename_parts) + ".csv"
    
    # Build full output path
    output_dir = Path("data/FE_train_data") / dataset_name
    output_path = output_dir / filename
    
    return output_path


# =========================
# CSV PROCESSING
# =========================

def process_csv(
    input_csv: str | Path,
    task_type: str,
    selection_method: str,
    top_k: int | None,
    use_log1p: bool,
    random_seed: int,
) -> None:
    """
    Run FE pipeline on a single CSV.
    
    Args:
        input_csv: Path to input CSV file
        task_type: Task type ("classification" | "regression") - must be specified
        selection_method: Feature selection method
        top_k: Number of top features to select
        use_log1p: Whether to apply log1p transform
        random_seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(random_seed)

    input_path = Path(input_csv)
    
    # Auto-generate output path
    output_path = generate_output_path(
        input_csv=input_path,
        selection_method=selection_method,
        top_k=top_k,
        random_seed=random_seed,
        use_log1p=use_log1p,
    )

    print(f"\n=== Processing CSV ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    df = pd.read_csv(input_path)

    df_aug = run_feature_engineering_on_df(
        df=df,
        selection_method=selection_method,
        top_k=top_k,
        use_log1p=use_log1p,
        rng=rng,
        task_type=task_type,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(output_path, index=False)

    print("Done.")

# =========================
# MAIN ENTRYPOINT
# =========================

if __name__ == "__main__":
    process_csv(
        input_csv=INPUT_CSV,
        task_type=TASK_TYPE,
        selection_method=FEATURE_SELECTION_METHOD,
        top_k=TOP_K_NUMERIC_FEATURES,
        use_log1p=USE_LOG1P,
        random_seed=RANDOM_SEED,
    )
