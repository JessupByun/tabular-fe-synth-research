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

The `feature_engineering.py` script supports both single-file and batch processing modes.

#### Configuration

Edit the configuration section at the top of `feature_engineering.py`:

```python
# Processing mode: "single" or "batch"
MODE = "batch"

# Feature selection method: "mi", "tree", or "random"
FEATURE_SELECTION_METHOD = "mi"

# Number of top features to select
TOP_K_NUMERIC_FEATURES = 5

# Transform tier: 0 (No FE) or 1 (Full FE)
TRANSFORM_TIER = 1

# Optional log1p transform (ablation)
USE_LOG1P = False
```

#### Running the Pipeline

**Batch Mode** (processes all datasets):
```bash
python feature_engineering.py
```

**Single File Mode**:
1. Set `MODE = "single"`
2. Configure `SINGLE_INPUT_CSV`, `SINGLE_OUTPUT_CSV`, and `SINGLE_METADATA_JSON`
3. Run: `python feature_engineering.py`

#### Output

For each dataset, the pipeline generates:
- `train_fe.csv`: Augmented training data with engineered features
- `train_fe_meta.json`: Metadata including:
  - Selected features and their scores
  - Engineered column names
  - Configuration parameters
  - Task type (classification/regression)

### Example Workflow

1. **Run feature engineering**:
   ```bash
   python feature_engineering.py
   ```

2. **Train generators** on `data/synthetic_data/feature_eng_data/<dataset>/train_fe.csv`

3. **Project back to original schema** by dropping engineered columns (columns ending in `_sq`, `_x_`, `_log1p`)

4. **Evaluate** synthetic data on original feature space

## Reproducibility

- Random seed: `42` (configurable via `RANDOM_SEED`)
- All configurations saved in metadata JSON files
- Deterministic transforms ensure consistent results

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost` (optional, falls back to RandomForest if not available)

## Contributing

This is a research project in progress. For questions or collaboration, please contact the project maintainer.

## License

See `LICENSE` file for details.