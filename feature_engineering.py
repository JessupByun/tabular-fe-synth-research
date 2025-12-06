"""
Feature engineering pipeline for tabular data.

- Supports single-CSV processing and batch processing over many datasets.
- Performs feature selection (MI / variance / random).
- Applies deterministic transforms (squares, pairwise sums/products, optional log1p).
- Writes augmented CSVs plus metadata JSON with full configuration and selected columns.
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

# ----------------------------------------------------------------------------
# Processing Mode
# ----------------------------------------------------------------------------
# Options: "single" | "batch"
#   - "single": Process one CSV file
#   - "batch":  Process all datasets in a directory structure
MODE = "batch"

# ----------------------------------------------------------------------------
# Single-File Mode Settings (only used when MODE == "single")
# ----------------------------------------------------------------------------
SINGLE_INPUT_CSV = "path/to/your/train.csv"           # Input CSV file path
SINGLE_OUTPUT_CSV = "path/to/your/train_fe.csv"       # Output CSV file path
SINGLE_METADATA_JSON = "path/to/your/train_fe_meta.json"  # Metadata JSON file path

# Task type for single-file mode (ONLY used when MODE == "single")
# Options: "classification" | "regression" | None
#   - "classification": Force classification task type
#   - "regression":     Force regression task type
#   - None:             Auto-detect task type using detect_task_type() helper
# Note: Batch processing mode always auto-detects task type per dataset,
#       regardless of this setting.
SINGLE_TASK_TYPE = None

# ----------------------------------------------------------------------------
# Batch Mode Settings (only used when MODE == "batch")
# ----------------------------------------------------------------------------
# Directory structure:
#   Input:  <REAL_ROOT_DIR>/<dataset_name>/train/*.csv
#   Output: <OUTPUT_ROOT_DIR>/<dataset_name>/train_fe.csv
#           <OUTPUT_ROOT_DIR>/<dataset_name>/train_fe_meta.json
REAL_ROOT_DIR = "data/real_data"                          # Root directory containing dataset folders
OUTPUT_ROOT_DIR = "data/synthetic_data/feature_eng_data"  # Root directory for output files

# ----------------------------------------------------------------------------
# Feature Selection Settings
# ----------------------------------------------------------------------------
# Method options: "mi" | "variance" | "random"
#   - "mi":       Select features by mutual information with target
#   - "variance": Select features by variance (highest variance first)
#   - "random":   Randomly select features
FEATURE_SELECTION_METHOD = "mi"

# Number of top features to select
# Options: int (positive number) | None
#   - int:  Select top K features (default is 5)
#   - None: Select all numeric features
TOP_K_NUMERIC_FEATURES = 5

# Random seed for reproducibility
# Options: int (any integer)
#   - Used for random feature selection and sklearn's MI calculations
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Feature Transform Settings
# ----------------------------------------------------------------------------
# Transform tier determines which transforms are applied
# Options: 0 | 1 | 2
#   - 0: No transforms (original features only)
#   - 1: Per-feature squares only (adds x_sq for each feature x)
#   - 2: Per-feature squares + pairwise products (adds x_sq and x_y for pairs)
TRANSFORM_TIER = 2

# Apply log1p transform to positive values
# Options: True | False
#   - True:  Adds x_log1p column for each feature (only for positive values)
#   - False: Skip log1p transform
USE_LOG1P = False
# Include pairwise sums in addition to products (only applies when TRANSFORM_TIER >= 2)
# Options: True | False
#   - True:  Adds x_plus_y columns for feature pairs
#   - False: Only add pairwise products (x_y), not sums
INCLUDE_PAIRWISE_SUMS = False
# “We restrict deterministic FE to second-order polynomial expansions over the top-k numeric features (squares and pairwise products).”

# ----------------------------------------------------------------------------
# Target Column Settings
# ----------------------------------------------------------------------------
# Target column name in the dataset
# Options: str (column name) | None
#   - str:  Use the specified column as the target
#   - None: Automatically use the rightmost column as the target
TARGET_COLUMN_NAME = None

# =========================
# IMPORTS
# =========================

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# =========================
# CORE HELPERS: TARGET & TASK
# =========================

def infer_target_column(df: pd.DataFrame, target_col: str | None) -> str:
    """If target_col is not given, assume the rightmost column is the target."""
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        return target_col
    return df.columns[-1]

def detect_task_type(y: pd.Series) -> str:
    """
    Decide whether the target is for classification or regression.

    How it works:
    - Counts the number of unique (non-NA) values in the series.
    - If the dtype is 'object' (i.e., likely categorical/text), or
      the number of unique values is small (<= min(20, len(y)//10)), 
      it infers classification.
    - Otherwise, it infers regression.

    In other words, if the target is categorical, or has very few unique values for its size (like labels), it's classification.
    If it's a continuous/real-valued variable with many unique values, it's regression.
    """
    nunique = y.nunique(dropna=True)
    # Classification if categorical dtype or few unique classes for the number of samples
    if y.dtype == "object" or nunique <= min(20, len(y) // 10):
        return "classification"
    return "regression"

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
    task_type: str | None = None,
) -> pd.Series:
    """
    Rank numeric features by mutual information with the target.
    
    Args:
        df: DataFrame containing features and target
        numeric_cols: List of numeric column names to rank
        target_col: Name of the target column
        task_type: Optional task type ("classification" | "regression" | None).
                   If None, auto-detects using detect_task_type().
    
    Returns:
        Series of MI scores sorted in descending order
    """
    if not numeric_cols:
        raise ValueError("No numeric features found to rank")

    y = df[target_col]
    X = df[numeric_cols]

    # Use provided task_type or auto-detect
    if task_type is None:
        task_type = detect_task_type(y)
    elif task_type not in ("classification", "regression"):
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")

    if task_type == "classification":
        mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    else:
        mi_scores = mutual_info_regression(X, y, discrete_features=False, random_state=0)

    mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)
    return mi_series


def rank_features_by_variance(df: pd.DataFrame, numeric_cols: List[str]) -> pd.Series:
    """Rank numeric features by variance."""
    if not numeric_cols:
        raise ValueError("No numeric features found to rank")
    var_series = df[numeric_cols].var().sort_values(ascending=False)
    return var_series


def select_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    method: str,
    top_k: int | None,
    rng: np.random.Generator,
    task_type: str | None = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select features according to the chosen method.

    Args:
        df: DataFrame containing features and target
        numeric_cols: List of numeric column names
        target_col: Name of the target column
        method: Feature selection method
            - "mi"       : mutual information with target
            - "variance" : variance-based ranking
            - "random"   : random subset of numeric_cols
        top_k: Number of top features to select (None for all)
        rng: Random number generator
        task_type: Optional task type ("classification" | "regression" | None).
                   Only used when method == "mi". If None, auto-detects.

    Returns:
        selected_cols, scores_dict
    """
    method = method.lower()
    scores: pd.Series

    if method == "mi":
        scores = rank_features_by_mi(df, numeric_cols, target_col, task_type=task_type)
    elif method == "variance":
        scores = rank_features_by_variance(df, numeric_cols)
    elif method == "random":
        if not numeric_cols:
            raise ValueError("No numeric features found for random selection")
        # Assign dummy scores (all equal) for completeness
        scores = pd.Series(1.0, index=numeric_cols)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

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

def get_transform_flags_from_tier(tier: int) -> Dict[str, bool]:
    """
    Map a transform tier to boolean flags.

    Tier 0: no transforms
    Tier 1: per-feature squares only
    Tier 2: per-feature squares + pairwise products (and optionally sums via INCLUDE_PAIRWISE_SUMS)
    """
    if tier == 0:
        return {
            "use_squares": False,
            "use_pairwise_products": False,
            "use_pairwise_sums": False,
        }
    if tier == 1:
        return {
            "use_squares": True,
            "use_pairwise_products": False,
            "use_pairwise_sums": False,
        }
    if tier == 2:
        return {
            "use_squares": True,
            "use_pairwise_products": True,
            "use_pairwise_sums": INCLUDE_PAIRWISE_SUMS,
        }
    raise ValueError(f"Unknown transform tier: {tier}")


def apply_deterministic_transforms(
    df: pd.DataFrame,
    cols: List[str],
    use_squares: bool,
    use_pairwise_sums: bool,
    use_pairwise_products: bool,
    use_log1p: bool,
    log_eps: float = 1e-6,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add engineered columns to df based on selected numeric columns.

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

    # Pairwise transforms
    n = len(cols)
    if use_pairwise_sums or use_pairwise_products:
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = cols[i], cols[j]

                if use_pairwise_sums:
                    new_col = f"{c1}_plus_{c2}"
                    df[new_col] = df[c1] + df[c2]
                    engineered_cols.append(new_col)

                if use_pairwise_products:
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
    transform_tier: int,
    use_log1p: bool,
    target_col_name: str | None,
    rng: np.random.Generator,
    task_type: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full FE pipeline on a single DataFrame.

    Args:
        df: DataFrame to process
        selection_method: Feature selection method ("mi", "variance", or "random")
        top_k: Number of top features to select (None for all)
        transform_tier: Transform tier (0, 1, or 2)
        use_log1p: Whether to apply log1p transform
        target_col_name: Name of target column (None to auto-detect)
        rng: Random number generator
        task_type: Optional task type ("classification" | "regression" | None).
                   If None, auto-detects using detect_task_type().

    Returns:
        df_augmented, metadata_dict
    """
    target_col = infer_target_column(df, target_col_name)
    numeric_cols = select_numeric_features(df, target_col)

    # Detect task type if not provided
    if task_type is None:
        y = df[target_col]
        task_type = detect_task_type(y)
        task_type_source = "auto-detected"
    else:
        if task_type not in ("classification", "regression"):
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")
        task_type_source = "manual"

    print(f"Using target column: {target_col}")
    print(f"Task type: {task_type} ({task_type_source})")
    print(f"Found {len(numeric_cols)} numeric feature(s): {numeric_cols}")

    selected_cols, scores_dict = select_features(
        df=df,
        numeric_cols=numeric_cols,
        target_col=target_col,
        method=selection_method,
        top_k=top_k,
        rng=rng,
        task_type=task_type if selection_method == "mi" else None,
    )

    print(f"Selected {len(selected_cols)} feature(s) for engineering using '{selection_method}': {selected_cols}")

    transform_flags = get_transform_flags_from_tier(transform_tier)

    df_aug, engineered_cols = apply_deterministic_transforms(
        df=df,
        cols=selected_cols,
        use_squares=transform_flags["use_squares"],
        use_pairwise_sums=transform_flags["use_pairwise_sums"],
        use_pairwise_products=transform_flags["use_pairwise_products"],
        use_log1p=use_log1p,
    )

    print(f"Added {len(engineered_cols)} engineered column(s).")

    metadata = {
        "target_col": target_col,
        "task_type": task_type,
        "task_type_source": task_type_source,
        "all_numeric_cols": numeric_cols,
        "selected_numeric_for_fe": selected_cols,
        "engineered_cols": engineered_cols,
        "config": {
            "selection_method": selection_method,
            "top_k": top_k,
            "transform_tier": transform_tier,
            "use_log1p": use_log1p,
            "random_seed": int(rng.bit_generator.state["state"]["state"]) if hasattr(rng.bit_generator, "state") else None,
        },
        "scores": scores_dict,
    }

    return df_aug, metadata


# =========================
# SINGLE CSV WRAPPER
# =========================

def process_single_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    metadata_json: str | Path,
    task_type: str | None = None,
) -> None:
    """
    Run FE pipeline on a single CSV.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        metadata_json: Path to metadata JSON file
        task_type: Optional task type ("classification" | "regression" | None).
                   If None, auto-detects using detect_task_type().
    """
    rng = np.random.default_rng(RANDOM_SEED)

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    metadata_json = Path(metadata_json)

    print(f"\n=== Processing single CSV ===")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Meta:   {metadata_json}")

    df = pd.read_csv(input_csv)

    df_aug, metadata = run_feature_engineering_on_df(
        df=df,
        selection_method=FEATURE_SELECTION_METHOD,
        top_k=TOP_K_NUMERIC_FEATURES,
        transform_tier=TRANSFORM_TIER,
        use_log1p=USE_LOG1P,
        target_col_name=TARGET_COLUMN_NAME,
        rng=rng,
        task_type=task_type,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    df_aug.to_csv(output_csv, index=False)
    with metadata_json.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")


# =========================
# BATCH WRAPPER
# =========================

def find_train_csv(train_dir: Path) -> Path | None:
    """Find a train CSV inside the given train/ directory."""
    if not train_dir.is_dir():
        return None
    csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        return None
    if len(csv_files) > 1:
        print(f"Warning: multiple CSVs in {train_dir}, using {csv_files[0].name}")
    return csv_files[0]


def process_batch_datasets(real_root: str | Path, output_root: str | Path) -> None:
    """
    Run FE pipeline over all datasets under real_root.
    
    Note: Task type is always auto-detected per dataset using detect_task_type().
          The SINGLE_TASK_TYPE configuration setting is not used in batch mode.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    real_root = Path(real_root)
    output_root = Path(output_root)

    print(f"\n=== Batch processing datasets ===")
    print(f"Real root:   {real_root}")
    print(f"Output root: {output_root}")

    dataset_dirs = sorted(d for d in real_root.iterdir() if d.is_dir())
    print(f"Found {len(dataset_dirs)} dataset folder(s).")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        train_dir = dataset_dir / "train"
        train_csv = find_train_csv(train_dir)

        if train_csv is None:
            print(f"Skipping dataset '{dataset_name}': no train CSV found in {train_dir}")
            continue

        print(f"\n--- Dataset: {dataset_name} ---")
        print(f"Train CSV: {train_csv}")

        df = pd.read_csv(train_csv)

        # Note: task_type is not passed, so it defaults to None and auto-detects
        df_aug, metadata = run_feature_engineering_on_df(
            df=df,
            selection_method=FEATURE_SELECTION_METHOD,
            top_k=TOP_K_NUMERIC_FEATURES,
            transform_tier=TRANSFORM_TIER,
            use_log1p=USE_LOG1P,
            target_col_name=TARGET_COLUMN_NAME,
            rng=rng,
        )

        # Output paths: <OUTPUT_ROOT>/<dataset_name>/train_fe.csv and train_fe_meta.json
        dataset_out_dir = output_root / dataset_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        out_csv = dataset_out_dir / "train_fe.csv"
        out_meta = dataset_out_dir / "train_fe_meta.json"

        df_aug.to_csv(out_csv, index=False)
        with out_meta.open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved augmented train to {out_csv}")
        print(f"Saved metadata to      {out_meta}")


# =========================
# MAIN ENTRYPOINT
# =========================

if __name__ == "__main__":
    if MODE == "single":
        process_single_csv(
            input_csv=SINGLE_INPUT_CSV,
            output_csv=SINGLE_OUTPUT_CSV,
            metadata_json=SINGLE_METADATA_JSON,
            task_type=SINGLE_TASK_TYPE,
        )
    elif MODE == "batch":
        process_batch_datasets(
            real_root=REAL_ROOT_DIR,
            output_root=OUTPUT_ROOT_DIR,
        )
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
