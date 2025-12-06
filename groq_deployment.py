import os
import json
import pandas as pd
import time
from io import StringIO
from dotenv import load_dotenv
import groq

# Retrieve the Groq API key and instantiate client
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")
client = groq.Groq(api_key=api_key)

# =============================
# Configuration (edit here)
# =============================
CONFIG = {
    "dataset_name": "white-wine",  # Dataset folder name under REAL_DATA_BASE
    "generator_name": "llama",
    "model_name": "llama-3.3-70b-versatile",
    "model_temperature": 1.0,  # Higher -> more diverse synthetic samples
    "batch_size": 32,
    # Input/Output base directories (must match your local layout)
    "REAL_DATA_BASE": os.path.join("LTM_data", "LTM_real_data"),
    "SYNTHETIC_DATA_BASE": os.path.join(
        "LTM_data", "LTM_synthetic_data", "LTM_llama_synthetic_data"
    ),
}

# Derived path constants (do not edit unless you change structure)
REAL_DATA_BASE = CONFIG["REAL_DATA_BASE"]
SYNTHETIC_DATA_BASE = CONFIG["SYNTHETIC_DATA_BASE"]

# Prompt template includes dataset name, summary statistics, column names, and full CSV data.
prompt_template = (
    """
    System role: You are a tabular synthetic data generation model.

    Your goal is to produce data that mirrors the given examples in causal structure and feature and label distributions, while producing as diverse samples as possible.

    Context: Leverage your prior knowledge and in-context learning capabilities to generate realistic but diverse samples.
    Output the data in JSON format that will be intended for csv formatting.

    Dataset name: {dataset_name}
    Please output data with exactly these column names in this order: {col_names}
    Do not emit trailing commas or extra columns.  
    Here is the CSV of the full data: {data}
    Please generate {batch_size} rows of synthetic data for the dataset. 

    Treat the rightmost column as the target, and return your entire response as a JSON object with the key 'synthetic_data' containing a CSV string of the generated data.
    Do not include any additional text.
    """
)

#    Summary statistics and information about numerical and categorical columns: {summary_stats}

def get_summary_statistics(df):
    """
    Computes a comprehensive set of summary statistics for each column in the DataFrame.
    
    For numeric columns, it calculates:
      - mean, median, mode (first mode value if multiple), standard deviation, min, max,
      - 25th and 75th percentiles,
      - number of unique values.
      
    For non-numeric (categorical) columns, it calculates:
      - the number of unique values,
      - the most common value (mode),
      - and the full value counts as a dictionary.
    
    Returns:
        A JSON string representation of the summary statistics.
    """
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "mode": float(mode_val) if pd.notnull(mode_val) else None,
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "25%": float(df[col].quantile(0.25)),
                "75%": float(df[col].quantile(0.75)),
                "unique_count": int(df[col].nunique())
            }
        else:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            value_counts = df[col].value_counts().to_dict()
            value_counts = {str(k): int(v) for k, v in value_counts.items()}
            stats[col] = {
                "unique_count": int(df[col].nunique()),
                "mode": str(mode_val) if mode_val is not None else None,
                "value_counts": value_counts
            }
    return json.dumps(stats, indent=2)

def extract_required_n_from_filename(filename: str) -> int:
    """
    From something like "abalone--train--64-seed1.csv" or "iris--train--150.csv",
    extract the number after "--train--" and before "-seed" (if present).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    if "--train--" not in base:
        return None
    part = base.split("--train--", 1)[1]
    if "-seed" in part:
        num_str = part.split("-seed", 1)[0]
    else:
        num_str = part
    try:
        return int(num_str)
    except:
        return None

def generate_synthetic_data_llama(df, dataset_name, model_name, batch_size, model_temperature):
    """
    Generates synthetic data using the Groq API and an LLM model.
    
    Returns the synthetic CSV string if successful, else None.
    """
    data_string = df.to_csv(index=False)
    col_names = ", ".join(df.columns)
    summary_stats = get_summary_statistics(df)

    prompt = prompt_template.format(
        data=data_string,
        dataset_name=dataset_name,
        col_names=col_names,
        summary_stats=summary_stats,
        batch_size=batch_size
    )
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            response_format={"type": "json_object"},
            temperature=model_temperature
        )
        if response.choices and len(response.choices) > 0:
            generated_text = response.choices[0].message.content
            parsed = json.loads(generated_text)
            return parsed.get("synthetic_data", None)
        else:
            print("No choices were returned in the response.")
            return None
    except Exception as e:
        print(f"Error generating data with model {model_name}: {e}")
        return None

def process_csv_file_llama(input_csv, output_csv, dataset_name, model_name, model_temperature, batch_size=200):
    """
    Loads, shuffles, batches, generates, then validates row count and re-prompts if needed.
    """
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    required_n = extract_required_n_from_filename(input_csv) or df.shape[0]

    synthetic_df_list = []
    for start in range(0, df.shape[0], batch_size):
        batch_df = df.iloc[start:start+batch_size]
        context_df = batch_df

        # keep trying until we get a valid, non-empty DataFrame
        while True:
            synthetic_csv = generate_synthetic_data_llama(
                batch_df, dataset_name, model_name, batch_size, model_temperature
            )

            if not synthetic_csv:
                print(f"[WARN] No synthetic data for batch starting at {start}, retrying…")
                time.sleep(1)
                continue

            try:
                batch_synthetic = pd.read_csv(StringIO(synthetic_csv))
            except Exception as e:
                print(f"[ERROR] Couldn't parse CSV for batch at {start}: {e}, retrying…")
                time.sleep(1)
                continue

            if batch_synthetic.empty:
                print(f"[WARN] Empty DataFrame for batch at {start}, retrying…")
                time.sleep(1)
                continue

            # success!
            synthetic_df_list.append(batch_synthetic)
            break

    if not synthetic_df_list:
        print("No synthetic data generated for file:", input_csv)
        return

    synthetic_df = pd.concat(synthetic_df_list, ignore_index=True)

    # refill loop: only retry up to 3 times, and require actual new rows
    attempts = 0
    max_attempts = 5
    while synthetic_df.shape[0] < required_n and attempts < max_attempts:
        needed = required_n - synthetic_df.shape[0]
        print(f"[INFO] Reprompting for {needed} more rows (attempt {attempts+1}/{max_attempts})")
        extra_csv = generate_synthetic_data_llama(
            context_df, dataset_name, model_name, min(needed, batch_size), model_temperature
        )
        if not extra_csv:
            print("[WARN] No data returned on reprompt, aborting.")
            break
        try:
            extra_df = pd.read_csv(StringIO(extra_csv))
        except Exception as e:
            print(f"[ERROR] Couldn't parse reprompt batch: {e}")
            break
        if extra_df.empty:
            print("[WARN] Empty DataFrame on reprompt, aborting.")
            break

        before = synthetic_df.shape[0]
        synthetic_df = pd.concat([synthetic_df, extra_df], ignore_index=True)
        if synthetic_df.shape[0] == before:
            print("[WARN] Reprompt did not increase row count, aborting.")
            break
        attempts += 1

    # truncate to exactly required_n (or warn if still off)
    if synthetic_df.shape[0] > required_n:
        synthetic_df = synthetic_df.iloc[:required_n]
    if synthetic_df.shape[0] != required_n:
        print(f"[WARNING] Final row count {synthetic_df.shape[0]} != required {required_n}")

    synthetic_df.to_csv(output_csv, index=False)
    print(f"[INFO] Synthetic data saved to {output_csv}")

def process_dataset_llama(dataset_name, generator_name, model_name, model_temperature, batch_size=200):
    """
    Processes all CSVs in the train folder, then runs the validation script.
    """
    real_data_path = os.path.join(REAL_DATA_BASE, dataset_name)
    train_folder = os.path.join(real_data_path, "train")
    if not os.path.isdir(train_folder):
        print(f"[ERROR] Train folder not found: {train_folder}")
        return

    synthetic_folder = os.path.join(
        SYNTHETIC_DATA_BASE, f"synth_{dataset_name}"
    )
    os.makedirs(synthetic_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(train_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"[WARNING] No CSV files found in {train_folder}.")
        return

    for csv_file in csv_files:
        input_csv = os.path.join(train_folder, csv_file)
        base = os.path.splitext(csv_file)[0]
        output_csv = os.path.join(synthetic_folder, f"{base}_{generator_name}_default_0.csv")
        print(f"[INFO] Processing: {input_csv} -> {output_csv}")
        process_csv_file_llama(
            input_csv, output_csv,
            dataset_name, model_name, model_temperature,
            batch_size=batch_size
        )

    # Run validation
    try:
        from validate_synthetic_data import validate_synthetic_data, logger
    except ImportError as e:
        print(f"Error importing validation function: {e}")
        return

    args = {
        "real_data_dir": os.path.join(real_data_path, "train"),
        "synthetic_data_dir": synthetic_folder,
        "output_file": os.path.join(synthetic_folder, f"{dataset_name}_validation_results.json")
    }

    validation_results = validate_synthetic_data(
        args["real_data_dir"],
        args["synthetic_data_dir"]
    )

    passed = sum(1 for r in validation_results if r["validation_passed"])
    total = len(validation_results)
    logger.info(f"Validation complete: {passed}/{total} synthetic datasets passed all checks")

    for r in validation_results:
        if not r["validation_passed"]:
            logger.warning(f"Issues with {r['synthetic_file']}:")
            for issue in r["issues"]:
                logger.warning(f"  - {issue}")

    try:
        with open(args["output_file"], "w") as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {args['output_file']}")
    except Exception as e:
        logger.error(f"Error saving validation results: {e}")

def main():
    dataset_name = CONFIG["dataset_name"]
    generator_name = CONFIG["generator_name"]
    model_name = CONFIG["model_name"]
    model_temperature = CONFIG["model_temperature"]  # Leave as 1.0 for highest diversity
    batch_size = CONFIG["batch_size"]

    process_dataset_llama(
        dataset_name,
        generator_name=generator_name,
        model_name=model_name,
        model_temperature=model_temperature,
        batch_size=batch_size
    )

if __name__ == "__main__":
    main()