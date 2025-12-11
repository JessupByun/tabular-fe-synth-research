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
# Use an explicit timeout to avoid hanging requests.
client = groq.Groq(api_key=api_key, timeout=60)

# =============================
# Configuration (edit here)
# =============================
CONFIG = {
    "dataset_name": "abalone",
    "input_csv_path": "data/FE_train_data/abalone/FE_abalone_train_mi_k5_seed42.csv",
    "output_csv_path": "data/synthetic_data/feature_eng_data/llama_synthetic_data/synth_abalone/synth_FE_abalone_train_mi_k5_seed42.csv",
    # Explicit target row count for the synthetic dataset; fallback to input rows if None
    "target_rows": None,
    "generator_name": "llama",
    "model_name": "llama-3.3-70b-versatile",
    "model_temperature": 1.0,  # Higher -> more diverse synthetic samples
    "batch_size": 200,
    # Per-request timeout in seconds for Groq API calls
    "request_timeout": 60,
}

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

def generate_synthetic_data_llama(df, dataset_name, model_name, batch_size, model_temperature, request_timeout):
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
        print(f"[DEBUG] Calling Groq: rows={len(df)}, cols={len(df.columns)}, prompt_chars={len(prompt)}")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            response_format={"type": "json_object"},
            temperature=model_temperature,
            timeout=request_timeout,
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

def process_csv_file_llama(
    input_csv,
    output_csv,
    dataset_name,
    model_name,
    model_temperature,
    batch_size=200,
    target_rows=None,
    request_timeout=60,
):
    """
    Loads once, then repeatedly prompts until the synthetic row count reaches required_n.
    Any over-generation is truncated to required_n.
    """
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    required_n = target_rows if target_rows is not None else df.shape[0]

    collected = []
    attempts = 0
    max_attempts = 50

    while True:
        current_n = sum(len(chunk) for chunk in collected)
        if current_n >= required_n:
            break

        remaining = required_n - current_n
        prompt_rows = min(batch_size, remaining)
        context_df = df.sample(n=min(len(df), prompt_rows), random_state=42 + attempts)

        print(f"[INFO] Generating batch attempt {attempts+1} with target {prompt_rows} rows (remaining {remaining})")
        synthetic_csv = generate_synthetic_data_llama(
            context_df,
            dataset_name,
            model_name,
            prompt_rows,
            model_temperature,
            request_timeout,
        )

        if not synthetic_csv:
            print(f"[WARN] No synthetic data returned (attempt {attempts+1}), retrying…")
            attempts += 1
            if attempts >= max_attempts:
                print("[ERROR] Max attempts reached; aborting generation.")
                break
            time.sleep(1)
            continue

        try:
            batch_synthetic = pd.read_csv(StringIO(synthetic_csv))
        except Exception as e:
            print(f"[ERROR] Couldn't parse CSV (attempt {attempts+1}): {e}, retrying…")
            attempts += 1
            if attempts >= max_attempts:
                print("[ERROR] Max attempts reached; aborting generation.")
                break
            time.sleep(1)
            continue

        if batch_synthetic.empty:
            print(f"[WARN] Empty DataFrame returned (attempt {attempts+1}), retrying…")
            attempts += 1
            if attempts >= max_attempts:
                print("[ERROR] Max attempts reached; aborting generation.")
                break
            time.sleep(1)
            continue

        collected.append(batch_synthetic)
        attempts += 1

    if not collected:
        print("No synthetic data generated for file:", input_csv)
        return

    synthetic_df = pd.concat(collected, ignore_index=True)

    # truncate to exactly required_n (or warn if still off)
    if synthetic_df.shape[0] > required_n:
        synthetic_df = synthetic_df.iloc[:required_n]
    if synthetic_df.shape[0] != required_n:
        print(f"[WARNING] Final row count {synthetic_df.shape[0]} != required {required_n}")

    synthetic_df.to_csv(output_csv, index=False)
    print(f"[INFO] Synthetic data saved to {output_csv}")

def main():
    dataset_name = CONFIG["dataset_name"]
    input_csv_path = CONFIG["input_csv_path"]
    output_csv_path = CONFIG["output_csv_path"]
    target_rows = CONFIG["target_rows"]
    request_timeout = CONFIG["request_timeout"]
    generator_name = CONFIG["generator_name"]
    model_name = CONFIG["model_name"]
    model_temperature = CONFIG["model_temperature"]  # Leave as 1.0 for highest diversity
    batch_size = CONFIG["batch_size"]

    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    output_path = output_csv_path
    print(f"[INFO] Processing single CSV: {input_csv_path}")
    print(f"[INFO] Saving synthetic CSV to: {output_path}")

    process_csv_file_llama(
        input_csv=input_csv_path,
        output_csv=output_path,
        dataset_name=dataset_name,
        model_name=model_name,
        model_temperature=model_temperature,
        batch_size=batch_size,
        target_rows=target_rows,
        request_timeout=request_timeout,
    )

if __name__ == "__main__":
    main()