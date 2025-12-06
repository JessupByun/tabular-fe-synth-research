import os
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_from_directory(directory_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a directory into a dictionary of DataFrames.
    
    Args:
        directory_path: Path to the directory containing CSV files
        
    Returns:
        Dictionary mapping filenames (without extension) to pandas DataFrames
    """
    data_dict = {}
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist")
        return data_dict
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)
                data_dict[os.path.splitext(filename)[0]] = df
                logger.info(f"Loaded {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    
    return data_dict

def parse_synthetic_filename(filename: str) -> str:
    """
    Parse synthetic filename to extract original table name.
    Expected format: "originalTableName_synthesizerName_default_testIdx.csv"
    
    Args:
        filename: Synthetic data filename
        
    Returns:
        Original table name
    """
    parts = filename.split('_')
    if len(parts) < 3:
        logger.error(f"Invalid synthetic filename format: {filename}")
        return None
    
    # The original table name could contain underscores, so we need to be careful
    # Assuming the last three parts are synthesizerName, default, and testIdx
    original_table_name = '_'.join(parts[:-3])
    return original_table_name

def validate_synthetic_data(real_data_dir: str, synthetic_data_dir: str) -> List[Dict]:
    """
    Validate synthetic data against real data according to the specified rules.
    
    Args:
        real_data_dir: Directory containing real data CSV files
        synthetic_data_dir: Directory containing synthetic data CSV files
        
    Returns:
        List of validation results for each synthetic dataset
    """
    # Step 1: Load the data from training csv dir and synthetic csv dir
    real_data = load_data_from_directory(real_data_dir)
    synthetic_data = load_data_from_directory(synthetic_data_dir)
    
    if not real_data:
        logger.error("No real data loaded")
        return []
    
    if not synthetic_data:
        logger.error("No synthetic data loaded")
        return []
    
    validation_results = []
    
    # Process each synthetic dataset
    for synth_name, synth_df in synthetic_data.items():
        result = {
            "synthetic_file": synth_name,
            "validation_passed": True,
            "issues": []
        }
        
        # Step 2 & 3: Parse filename and find corresponding real data
        original_table_name = parse_synthetic_filename(synth_name)
        if not original_table_name or original_table_name not in real_data:
            result["validation_passed"] = False
            result["issues"].append(f"Could not find corresponding real data for {synth_name}")
            validation_results.append(result)
            continue
        
        real_df = real_data[original_table_name]
        
        # Step 4: Check column names match
        if list(real_df.columns) != list(synth_df.columns):
            result["validation_passed"] = False
            result["issues"].append("Column names do not match between real and synthetic data")
            missing_cols = set(real_df.columns) - set(synth_df.columns)
            extra_cols = set(synth_df.columns) - set(real_df.columns)
            if missing_cols:
                result["issues"].append(f"Missing columns in synthetic data: {missing_cols}")
            if extra_cols:
                result["issues"].append(f"Extra columns in synthetic data: {extra_cols}")
        
        # Step 5: Check for missing/inf values
        for col in synth_df.columns:
            if col in real_df.columns:
                # Only check for missing/inf if real data doesn't have them
                if not real_df[col].isna().any() and synth_df[col].isna().any():
                    result["validation_passed"] = False
                    result["issues"].append(f"Column {col} has missing values in synthetic data but not in real data")
                
                # Check for inf values only if column is numeric
                if pd.api.types.is_numeric_dtype(real_df[col]) and pd.api.types.is_numeric_dtype(synth_df[col]):
                    if not np.isinf(real_df[col].replace([np.inf, -np.inf], np.nan).dropna()).any() and \
                       np.isinf(synth_df[col].replace([np.inf, -np.inf], np.nan).dropna()).any():
                        result["validation_passed"] = False
                        result["issues"].append(f"Column {col} has infinite values in synthetic data but not in real data")
        
        # Step 6: Check for empty rows/columns
        if synth_df.shape[0] == 0:
            result["validation_passed"] = False
            result["issues"].append("Synthetic data has no rows")
        
        for col in synth_df.columns:
            if synth_df[col].count() == 0:
                result["validation_passed"] = False
                result["issues"].append(f"Column {col} is empty in synthetic data")
        
        # Step 7: Check data types
        for col in synth_df.columns:
            if col in real_df.columns:
                real_dtype = real_df[col].dtype
                synth_dtype = synth_df[col].dtype
                
                # Compare basic dtype categories (numeric, object, etc.)
                if pd.api.types.is_numeric_dtype(real_dtype) != pd.api.types.is_numeric_dtype(synth_dtype) or \
                   pd.api.types.is_datetime64_dtype(real_dtype) != pd.api.types.is_datetime64_dtype(synth_dtype) or \
                   pd.api.types.is_categorical_dtype(real_dtype) != pd.api.types.is_categorical_dtype(synth_dtype):
                    result["validation_passed"] = False
                    result["issues"].append(f"Column {col} has different data type in synthetic data ({synth_dtype}) vs real data ({real_dtype})")
        
        validation_results.append(result)
    
    return validation_results

def main():
    """Main function to run the validation"""
    import argparse

    # python validate_synthetic_data.py --real_data_dir test_real_data --synthetic_data_dir test_synthetic_data --output_file check_synthetic_data.json
    
    parser = argparse.ArgumentParser(description='Validate synthetic data against real data')
    parser.add_argument('--real_data_dir', required=True, help='Directory containing real data CSV files')
    parser.add_argument('--synthetic_data_dir', required=True, help='Directory containing synthetic data CSV files')
    parser.add_argument('--output_file', help='Path to save validation results (JSON format)')
    
    args = parser.parse_args()
    
    validation_results = validate_synthetic_data(args.real_data_dir, args.synthetic_data_dir)
    
    # Print summary
    passed = sum(1 for result in validation_results if result["validation_passed"])
    total = len(validation_results)
    
    logger.info(f"Validation complete: {passed}/{total} synthetic datasets passed all checks")
    
    for result in validation_results:
        if not result["validation_passed"]:
            logger.warning(f"Issues with {result['synthetic_file']}:")
            for issue in result["issues"]:
                logger.warning(f"  - {issue}")
    
    # Save results if output file specified
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {args.output_file}")

if __name__ == "__main__":
    main()