#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from synth_mia.attackers import (
    gen_lra, dcr, dpi, logan, dcr_diff, domias,
    mc, density_estimate, local_neighborhood, classifier
)
from synth_mia.utils import tabular_preprocess

def get_attackers():
    return [
        gen_lra(hyper_parameters={"closest_compare_n": 1}),
        dcr(), dpi(), logan(), dcr_diff(), domias(),
        mc(), density_estimate(hyper_parameters={"method": "kde"}),
        local_neighborhood(), classifier(),
    ]

def compute_rare_masks(real_df: pd.DataFrame,
                       mem_df: pd.DataFrame,
                       non_mem_df: pd.DataFrame,
                       bin_count: int = 10,
                       threshold: float = 0.05):
    """
    1) Bin each numeric column of real_df into `bin_count` equal-width bins.
    2) Drop empty bins, then mark bins with freq < threshold as 'rare'.
       If none qualify, fall back to the single least-populated bin.
    3) Apply those same bins & rare‐bin sets to mem_df and non_mem_df.
    Returns:
      mem_mask, non_mem_mask (np.arrays of bools) and list of rare_cols.
    """
    bin_edges = {}
    rare_bins = {}
    rare_cols = []

    # 1) Determine rare bins on full real_df
    for col in real_df.select_dtypes(include=np.number).columns:
        cuts = pd.cut(real_df[col], bins=bin_count, include_lowest=True)
        freqs = cuts.value_counts(normalize=True).sort_index()
        freqs = freqs[freqs > 0]  # drop empty

        # find bins under threshold
        under = freqs[freqs < threshold].index.tolist()
        if not under:
            # fallback: pick the single smallest-frequency bin
            under = [freqs.idxmin()]

        bin_edges[col] = cuts.cat.categories
        rare_bins[col] = set(under)
        rare_cols.append(col)

    # helper to mark rows
    def mark(df):
        mask = pd.Series(False, index=df.index)
        for col, edges in bin_edges.items():
            b = pd.cut(df[col], bins=edges, include_lowest=True)
            mask |= b.isin(rare_bins[col])
        return mask

    mem_mask     = mark(mem_df).to_numpy()
    non_mem_mask = mark(non_mem_df).to_numpy()
    return mem_mask, non_mem_mask, rare_cols

def process_dataset_genmia(dataset_name: str, generator_name: str):
    base = dataset_name.split("_not_ablation")[0].split("_ablation_")[0]
    train_folder = os.path.join("LTM_data", "LTM_real_data", base, "train")
    test_folder  = os.path.join("LTM_data", "LTM_real_data", base, "test")
    synth_folder = os.path.join(
        "LTM_data", "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )
    output_root = os.path.join("LTM_evaluation", "LTM_Gen_MIA", generator_name, dataset_name)
    os.makedirs(output_root, exist_ok=True)

    test_files = [f for f in os.listdir(test_folder) if f.endswith(".csv")]
    assert len(test_files) == 1, f"Expected one CSV in {test_folder}"
    non_mem_df = pd.read_csv(os.path.join(test_folder, test_files[0]))

    attackers = get_attackers()

    for train_fname in sorted(os.listdir(train_folder)):
        if not train_fname.endswith(".csv"):
            continue

        split_name = os.path.splitext(train_fname)[0]
        real_df     = pd.read_csv(os.path.join(train_folder, train_fname))
        mem_df, ref_df = train_test_split(real_df, test_size=0.5, random_state=42)
        mem_df   = mem_df.reset_index(drop=True)
        ref_df   = ref_df.reset_index(drop=True)

        synth_path = os.path.join(synth_folder, f"{split_name}_{generator_name}_default_0.csv")
        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synthetic for {split_name}, skipping.")
            continue
        synth_df = pd.read_csv(synth_path)

        mem_arr, non_mem_arr, synth_arr, ref_arr, _ = tabular_preprocess(
            mem_df, non_mem_df, synth_df, ref_df,
            fit_target='synth', categorical_encoding='one-hot'
        )

        # compute rare‐row masks & record cols
        mem_mask, non_mem_mask, rare_cols = compute_rare_masks(real_df, mem_df, non_mem_df)
        full_mask = np.concatenate([mem_mask, non_mem_mask])

        print(f"[DEBUG] {split_name} rare rows → mem: {mem_mask.sum()}, "
              f"non_mem: {non_mem_mask.sum()}, total: {full_mask.sum()}")
        print(f"[DEBUG] {split_name} columns with rare bins: {rare_cols}")

        records = {}
        for attacker in attackers:
            try:
                scores, labels = attacker.attack(mem_arr, non_mem_arr, synth_arr, ref_arr)

                # overall AUC
                try:
                    overall_auc = roc_auc_score(labels, scores)
                except ValueError:
                    overall_auc = np.nan

                # rare‐only AUC
                if full_mask.sum() > 1 and np.unique(labels[full_mask]).size == 2:
                    rare_auc = roc_auc_score(labels[full_mask], scores[full_mask])
                else:
                    rare_auc = np.nan

                records[attacker.name] = {
                    "ROC_AUC":            float(overall_auc),
                    "Rare_Class_ROC_AUC": float(rare_auc),
                    "Mem_Rare_Count":     int(mem_mask.sum()),
                    "NonMem_Rare_Count":  int(non_mem_mask.sum())
                }
            except Exception as e:
                print(f"[ERROR] {attacker.name} on {split_name}: {e}")
                records[attacker.name] = {
                    "ROC_AUC": np.nan,
                    "Rare_Class_ROC_AUC": np.nan,
                    "Mem_Rare_Count": 0,
                    "NonMem_Rare_Count": 0
                }

        df_out = pd.DataFrame.from_dict(records, orient='index')
        df_out.index.name = 'Attacker'
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        df_out.to_csv(os.path.join(split_dir, "mia_results.csv"))
        print(f"[INFO] Saved MIA results to {split_dir}/mia_results.csv")

def main():
    generator = "llama"
    dataset_names = [
        "concrete-compressive-strength_not_ablation",
        "concrete-compressive-strength_ablation_batchsize",
        "concrete-compressive-strength_ablation_temp0.1",
        "concrete-compressive-strength_ablation_temp0.5",
        "concrete-compressive-strength_ablation_summarystats",
    ]

    for ds in dataset_names:
        print(f"\n=== Processing dataset: {ds} ===")
        try:
            process_dataset_genmia(ds, generator)
        except Exception as e:
            print(f"[ERROR] Failed on {ds}: {e}")

if __name__ == "__main__":
    main()
