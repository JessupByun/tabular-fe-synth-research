# Feature Engineering for Synthetic Tabular Data Generation Research

## Research Question

**How does feature engineering (FE) applied to real data before training synthetic data generators affect the utility, fidelity, diversity, and privacy of the generated synthetic data?**

This project investigates whether applying supervised feature engineering to real tabular data prior to generator training improves the quality of synthetic data.

## Experimental Design

### Overview

We apply feature engineering as the **only intervention** on real data before training synthetic data generators. After generation, engineered features are projected back to the original schema before evaluation, ensuring fair comparison with baseline methods.

### Pipeline

1. **Feature Selection**: Select top-k numeric features using supervised methods
2. **Feature Engineering**: Apply deterministic polynomial transforms (squares + pairwise products)
3. **Generator Training**: Train synthetic data generators on the augmented data
4. **Projection**: Drop engineered columns to restore original schema
5. **Evaluation**: Assess utility, fidelity, diversity, and privacy on original feature space

### Feature Selection

**Core Method:**
- **Mutual Information (MI)**: Select top-k numeric features (k=5) based on mutual information with the target variable
- Task-aware: Uses `mutual_info_classif` for classification and `mutual_info_regression` for regression

**Ablations:**
- **Tree/XGBoost Importance**: Select features using tree-based importance scores (XGBoost if available, otherwise RandomForest)
- **Random Selection**: Randomly select top-k features as a control baseline

**Note**: Variance-based selection is explicitly excluded from the design.

### Feature Transforms

**Main Experiment:**
- **No FE (Tier 0)**: Baseline with no feature engineering
- **Full FE (Tier 1)**: Per-feature squares + pairwise products on top-k numeric features
  - For each selected feature `x`: adds `x_sq = x²`
  - For each pair of selected features `(x, y)`: adds `x_x_y = x * y`
  - **Pairwise sums are excluded** per design

**Ablation:**
- **Full FE + log1p**: Adds `x_log1p = log(1 + x)` for positive values only (applied as an additional transform)

### Generator Suite

We evaluate the following synthetic data generators:

- **LLM-based**: LLaMA, GPT-4o-mini
- **Deep Learning**: TabPFN + TabDDPM, TabSyn, CTGAN, TVAE
- **Traditional**: SMOTE

### Datasets

We use a **15-dataset benchmark suite**, which includes:
- Mixed regression and classification tasks
- Numeric and categorical features
- Diverse domain applications

Datasets are located in `data/real_data/` with the following structure:
```
data/real_data/
  <dataset_name>/
    train/
      *.csv
    test/
      *.csv
```

### Evaluation Metrics

**Primary Focus:**
- **Fidelity**: How well synthetic data matches real data distributions
- **Utility**: Downstream task performance on synthetic data

**Secondary:**
- **Diversity**: Coverage and variety of generated samples
- **Privacy**: Resistance to membership inference attacks

## Project Status

🚧 **This project is currently ongoing and under active development.**

## Project Structure

```
feature_eng_research/
├── feature_engineering.py    # Main FE pipeline
├── data/
│   ├── real_data/            # Original datasets
│   └── synthetic_data/
│       ├── feature_eng_data/ # FE-augmented training data
│       └── no_feature_eng_data/  # Baseline (no FE)
├── evaluation/               # Evaluation scripts and results
└── synth_mia_script_updates/ # Privacy evaluation tools
```

## Usage

### Feature Engineering Pipeline

The `feature_engineering.py` script processes a single CSV file at a time.

#### Configuration

Edit the configuration section at the top of `feature_engineering.py`:

```python
# Input path
INPUT_CSV = "data/real_data/abalone/train/abalone_train.csv"
TASK_TYPE = "regression"  # Options: "classification" | "regression"

# Feature selection method: "mi", "tree", or "random"
FEATURE_SELECTION_METHOD = "mi"

# Number of top features to select
TOP_K_NUMERIC_FEATURES = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Optional log1p transform (ablation)
USE_LOG1P = False
```

**Important Notes:**
- **Target column**: Always the rightmost column in the DataFrame. Ensure your target is in the rightmost position.
- **Task type**: Must be manually specified as `"classification"` or `"regression"` for each dataset.
- **Output path**: Automatically generated based on input path and configuration. Format: `data/FE_train_data/{dataset_name}/FE_{dataset_name}_train_{method}_k{top_k}_seed{seed}[_log1p].csv`
- **Transforms**: Always applies Full FE (squares + pairwise products). No transform tier needed - baseline (no FE) data comes from train/test splits.

#### Running the Pipeline

1. Configure `INPUT_CSV` and `TASK_TYPE` in the script (output path is auto-generated)
2. Run: `python feature_engineering.py`

#### Output

The pipeline automatically generates an output CSV file with:
- **Path**: `data/FE_train_data/{dataset_name}/FE_{dataset_name}_train_{method}_k{top_k}_seed{seed}[_log1p].csv`
- **Content**: Original columns + engineered columns (identifiable by name patterns: `_sq`, `_x_`, `_log1p`)

**Example output paths:**
- `data/FE_train_data/abalone/FE_abalone_train_mi_k5_seed42.csv` (without log1p)
- `data/FE_train_data/abalone/FE_abalone_train_mi_k5_seed42_log1p.csv` (with log1p ablation)

### Example Workflow

1. **Run feature engineering**:
   ```bash
   python feature_engineering.py
   ```

2. **Train generators** on the output CSV

3. **Project back to original schema** by dropping engineered columns (columns ending in `_sq`, `_x_`, `_log1p`). Compare column names to the original holdout set to identify which columns to drop.

4. **Evaluate** synthetic data on original feature space

## Reproducibility

- Random seed: `42` (configurable via `RANDOM_SEED`)
- Deterministic transforms ensure consistent results
- Configuration encoded in output filenames (e.g., `FE_abalone_train_mi_5_tier1_seed42.csv`)

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost` (optional, falls back to RandomForest if not available)

## Contributing

This is a research project in progress. For questions or collaboration, please contact the project maintainer.

## License

See `LICENSE` file for details.