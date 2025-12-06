"""
Split a single CSV file into train and test sets (80/20 split).

Configure the paths below and run the script.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input CSV file path
INPUT_CSV = "data/real_data/abalone/abalone.csv"

# Output paths
TRAIN_OUTPUT_CSV = "data/real_data/abalone/train/abalone_train.csv"
TEST_OUTPUT_CSV = "data/real_data/abalone/test/abalone_test.csv"

# Split configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 20% for test, 80% for train


def split_csv(input_csv: str, train_output: str, test_output: str) -> None:
    """
    Split a CSV file into train and test sets.
    
    Args:
        input_csv: Path to input CSV file
        train_output: Path to output train CSV file
        test_output: Path to output test CSV file
    """
    input_path = Path(input_csv)
    train_path = Path(train_output)
    test_path = Path(test_output)
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    print(f"Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")
    
    # Perform train/test split
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"Train rows: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Test rows: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    # Create output directories if needed
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save train and test sets
    df_train.to_csv(train_path, index=False)
    print(f"Saved train set to: {train_output}")
    
    df_test.to_csv(test_path, index=False)
    print(f"Saved test set to: {test_output}")
    
    print("Done!")


def main():
    """Main entry point."""
    try:
        split_csv(INPUT_CSV, TRAIN_OUTPUT_CSV, TEST_OUTPUT_CSV)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

