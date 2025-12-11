import sys, types

# Stub out only the alpha_precision metric so it never gets imported
sys.modules['evaluator.metrics.alpha_precision'] = types.ModuleType('evaluator.metrics.alpha_precision')
sys.modules['evaluator.metrics.alpha_precision'].AlphaPrecision = type('AlphaPrecision', (), {})

# Now it’s safe to import the rest
from evaluator import EvaluationPipeline
import os, pandas as pd

# infer_column_types stays the same...
def infer_column_types(df: pd.DataFrame,
                       cat_threshold: int = 50,
                       rel_cardinality: float = 0.05) -> dict:
    col_types = {}
    n = len(df)
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_datetime64_any_dtype(ser):
            col_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(ser):
            uniq = ser.nunique(dropna=True)
            if uniq < cat_threshold or (uniq / n) < rel_cardinality:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numerical"
        else:
            col_types[col] = "categorical"
    return col_types

# anchor at the project root (where LTM_data lives)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

def process_dataset_alfred(dataset_name: str, generator_name: str):
    real_folder = os.path.join(PROJECT_ROOT, "LTM_data", "LTM_real_data", dataset_name, "train")
    synth_folder = os.path.join(
        PROJECT_ROOT,
        "LTM_data", "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )
    output_root = os.path.join(
        PROJECT_ROOT, "LTM_evaluation", "LTM_alfred_evaluation",
        generator_name, dataset_name
    )

    os.makedirs(output_root, exist_ok=True)

    real_files = sorted(f for f in os.listdir(real_folder) if f.endswith(".csv"))
    print(f"[INFO] Found {len(real_files)} real splits in {real_folder}")

    for real_fname in real_files:
        base = os.path.splitext(real_fname)[0]
        real_path  = os.path.join(real_folder, real_fname)
        synth_fname = f"{base}_{generator_name}_default_0.csv"
        synth_path  = os.path.join(synth_folder, synth_fname)

        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synth for {base}, skipping.")
            continue

        print(f"[INFO] Evaluating subset {base} ...")
        real_df  = pd.read_csv(real_path)
        synth_df = pd.read_csv(synth_path)

        col_types     = infer_column_types(real_df)
        target_column = real_df.columns[-1]

        config = {
            "target_column": target_column,
            "metadata": col_types,
            "holdout_seed": 42,
            "holdout_size": 0.2,
        }

        # here's the one‐per‐subset folder
        subset_output = os.path.join(output_root, base)
        os.makedirs(subset_output, exist_ok=True)

        pipeline = EvaluationPipeline(
            real_data=real_df,
            synth_data=synth_df,
            column_name_to_datatype=col_types,
            config=config,
            save_path=subset_output
        )
        pipeline.run_pipeline()
        print(f"[INFO] Finished {base} → {subset_output}")

def main():
    generator_name = "llama"
    dataset_names = [
    
    ]
    """
    "abalone", "airfoil-self-noise", "auction-verification", "brazilian-houses", "california-housing", "cars", "concrete-compressive-strength",
    "cps88wages", "cpu-activity", "diamonds", "energy-efficiency", "fifa", "forest-fires", "fps-benchmark", "geographical-origin-of-music", "grid-stability", "health-insurance",
    "kin8nm", "kings-county", "miami-housing", "Moneyball", "naval-propulsion-plant", "physiochemical-protein", "pumadyn32nh", "QSAR-fish-toxicity", "red-wine",
    "sarcos", "socmob", "solar-flare", "space-ga", "student-performance-por", "superconductivity", "video-transcoding", "wave-energy", "white-wine"
    """
    for ds in dataset_names:
        print(f"\n=== Processing dataset: {ds} ===")
        try:
            process_dataset_alfred(ds, generator_name)
        except Exception as e:
            print(f"[ERROR] Failed on {ds}: {e}")

if __name__ == "__main__":
    main()
