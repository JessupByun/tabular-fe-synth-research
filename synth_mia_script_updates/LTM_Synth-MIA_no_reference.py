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
    """Instantiate membership-inference attackers with default hyperparameters."""
    return [
        dcr(), 
        mc(), density_estimate(hyper_parameters={"method": "kde"}),
    ]


def process_dataset_genmia(dataset_name: str, generator_name: str):
    """
    For each CSV in LTM_data/LTM_real_data/{dataset_name}/train/:
      - split 50/50 into mem vs ref (seed=42)
      - load non_mem as the single CSV under LTM_data/LTM_real_data/{dataset_name}/test/
      - load matching synthetic CSV
      - run membership-inference attacks directly on the data
      - compute ROC AUC
      - save results
    """
    # Paths
    train_folder = os.path.join("LTM_data", "LTM_real_data", dataset_name, "train")
    test_folder = os.path.join("LTM_data", "LTM_real_data", dataset_name, "test")
    synth_folder = os.path.join(
        "LTM_data", "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )
    output_root = os.path.join("LTM_evaluation", "LTM_Gen_MIA_no_ref", generator_name, dataset_name)
    os.makedirs(output_root, exist_ok=True)

    # Check folders
    if not os.path.isdir(train_folder):
        raise NotADirectoryError(f"Train folder not found: {train_folder}")
    if not os.path.isdir(test_folder):
        raise NotADirectoryError(f"Test folder not found: {test_folder}")
    if not os.path.isdir(synth_folder):
        raise NotADirectoryError(f"Synthetic folder not found: {synth_folder}")

    # Identify the one test CSV
    test_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.csv')]
    if len(test_files) != 1:
        raise ValueError(f"Expected exactly one test CSV in {test_folder}, found {len(test_files)}")
    non_mem_path = os.path.join(test_folder, test_files[0])

    attackers = get_attackers()

    # Loop over each train CSV
    for train_fname in sorted(os.listdir(train_folder)):
        if not train_fname.lower().endswith('.csv'):
            continue

        base = os.path.splitext(train_fname)[0]
        real_path = os.path.join(train_folder, train_fname)
        synth_fname = f"{base}_{generator_name}_default_0.csv"
        synth_path = os.path.join(synth_folder, synth_fname)
        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synthetic for {base}: {synth_path}")
            continue

        # 1) Load real data
        df = pd.read_csv(real_path)
        
        # 2) Split into mem and ref sets
        #mem_df, ref_df = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)
        
        # 3) Load non_mem data
        non_mem_df = pd.read_csv(non_mem_path)
        
        # 4) Load synthetic data
        synth_df = pd.read_csv(synth_path)
        

        # Take preprocessing for synthetic data, apply to all dataframes, return np arrays
        mem, non_mem, synth, transformer = tabular_preprocess(df, non_mem_df, synth_df,  fit_target='synth', categorical_encoding='ordinal') #or categorical_encoding='ordinal')

        # DEBUG: check shapes before attacking
        print(f"[DEBUG] {base} shapes – mem: {mem.shape}, non_mem: {non_mem.shape}, synth: {synth.shape}")

        # 7) Run attacks and compute ROC AUC
        results = {}
        for attacker in attackers:
            try:
                scores, true_labels = attacker.attack(mem, non_mem, synth)
                # Evaluate the attack
                eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
                # Store results
                results[attacker.name] = eval_results
            except Exception as e:
                print(f"[ERROR] {attacker.name} on {base}: {e}")
                
        # 8) Save results
        df_out = pd.DataFrame.from_dict(results, orient='index')
        save_dir = os.path.join(output_root, base)
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(save_dir, 'mia_results.csv')
        df_out.to_csv(out_csv)
        print(f"[INFO] Saved MIA results to {out_csv}")


def main():
    generator_name = "CTGAN"
    dataset_names = [
        "abalone", "airfoil-self-noise", "auction-verification", "brazilian-houses", "california-housing", "cars", "concrete-compressive-strength",
    "cps88wages", "cpu-activity", "diamonds", "energy-efficiency", "fifa", "forest-fires", "grid-stability", "health-insurance",
    "kin8nm", "kings-county", "miami-housing", "Moneyball", "naval-propulsion-plant", "physiochemical-protein", "pumadyn32nh", "QSAR-fish-toxicity", "red-wine",
    "sarcos", "socmob", "solar-flare", "space-ga", "student-performance-por", "superconductivity", "video-transcoding", "wave-energy", "white-wine"
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
            process_dataset_genmia(ds, generator_name)
        except Exception as e:
            print(f"[ERROR] Failed on {ds}: {e}")

if __name__ == "__main__":
    main()