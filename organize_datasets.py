"""
Organize datasets in real_data folder:
- Each dataset gets its own folder
- Convert non-CSV files to CSV format
- Organize into train/test structure if applicable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Optional, Tuple
import re

# Try to import arff support
try:
    from scipy.io import arff  # type: ignore
    HAS_SCIPY_ARFF = True
    HAS_ARFF = False
except ImportError:
    try:
        import arff  # type: ignore
        HAS_ARFF = True
        HAS_SCIPY_ARFF = False
    except ImportError:
        HAS_ARFF = False
        HAS_SCIPY_ARFF = False
        print("Warning: No ARFF support. Install scipy or liac-arff to handle .arff files.")

REAL_DATA_DIR = Path("data/real_data")


def read_arff(file_path: Path) -> pd.DataFrame:
    """Read ARFF file and return DataFrame."""
    if HAS_SCIPY_ARFF:
        data, meta = arff.loadarff(str(file_path))
        df = pd.DataFrame(data)
        # Decode bytes to strings if needed
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        return df
    elif HAS_ARFF:
        with open(file_path, 'r') as f:
            arff_data = arff.load(f)
        df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
        return df
    else:
        raise ImportError("No ARFF library available. Install scipy or liac-arff.")


def read_data_file(file_path: Path) -> pd.DataFrame:
    """Read .data or .NNA file (usually CSV-like, try different delimiters)."""
    # Try common delimiters
    for delimiter in [',', '\t', ' ', ';']:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, header=None, low_memory=False)
            # Check if it looks reasonable (not too many columns, not all NaN)
            if df.shape[1] > 1 and df.shape[1] < 1000 and not df.isna().all().all():
                return df
        except Exception:
            continue
    
    # If all fail, try with no delimiter (space-separated)
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, low_memory=False)
        return df
    except Exception as e:
        raise ValueError(f"Could not parse data file {file_path}: {e}")


def read_excel(file_path: Path) -> pd.DataFrame:
    """Read Excel file."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        return df
    except ImportError:
        try:
            df = pd.read_excel(file_path, engine='xlrd')
            return df
        except Exception:
            # Try without specifying engine
            try:
                df = pd.read_excel(file_path)
                return df
            except Exception as e:
                raise ValueError(f"Could not read Excel file {file_path}: {e}. Install openpyxl or xlrd.")


def convert_to_csv(input_path: Path, output_path: Path) -> bool:
    """Convert various file formats to CSV."""
    suffix = input_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            # Already CSV, just copy
            shutil.copy2(input_path, output_path)
            return True
        elif suffix == '.arff':
            df = read_arff(input_path)
            df.to_csv(output_path, index=False)
            return True
        elif suffix == '.data':
            df = read_data_file(input_path)
            df.to_csv(output_path, index=False)
            return True
        elif suffix in ['.xlsx', '.xls']:
            df = read_excel(input_path)
            df.to_csv(output_path, index=False)
            return True
        elif suffix == '.nna':
            # .NNA files are usually tab-separated
            df = read_data_file(input_path)
            df.to_csv(output_path, index=False)
            return True
        else:
            print(f"  Warning: Unknown file format {suffix} for {input_path.name}")
            return False
    except Exception as e:
        print(f"  Error converting {input_path.name}: {e}")
        return False


def get_dataset_name_from_file(file_path: Path) -> str:
    """Extract dataset name from file path."""
    # Remove common suffixes and clean up
    name = file_path.stem
    # Remove common patterns
    name = re.sub(r'\.(data|train|test)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[_-]?(train|test|data)$', '', name, flags=re.IGNORECASE)
    # Clean up special characters
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name.lower() if name else file_path.stem.lower()


def organize_dataset_folder(dataset_dir: Path) -> None:
    """Organize files in a dataset folder into train/test structure."""
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Find all data files
    data_files = []
    for file_path in dataset_dir.iterdir():
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix in ['.csv', '.arff', '.data', '.xlsx', '.xls', '.nna']:
                data_files.append(file_path)
    
    if not data_files:
        return
    
    # Try to identify train/test files
    train_files = []
    test_files = []
    other_files = []
    
    for file_path in data_files:
        name_lower = file_path.name.lower()
        if 'test' in name_lower:
            test_files.append(file_path)
        elif 'train' in name_lower or 'data' in name_lower:
            train_files.append(file_path)
        else:
            other_files.append(file_path)
    
    # If we have both train and test, organize them
    if train_files and test_files:
        # Convert and move train files
        for file_path in train_files:
            output_csv = train_dir / f"{file_path.stem}.csv"
            if convert_to_csv(file_path, output_csv):
                if file_path.suffix.lower() != '.csv':
                    print(f"  Converted {file_path.name} -> train/{output_csv.name}")
        
        # Convert and move test files
        for file_path in test_files:
            output_csv = test_dir / f"{file_path.stem}.csv"
            if convert_to_csv(file_path, output_csv):
                if file_path.suffix.lower() != '.csv':
                    print(f"  Converted {file_path.name} -> test/{output_csv.name}")
    
    # If we only have one type or other files, put them in train
    elif train_files or other_files:
        all_files = train_files + other_files
        for file_path in all_files:
            output_csv = train_dir / f"{file_path.stem}.csv"
            if convert_to_csv(file_path, output_csv):
                if file_path.suffix.lower() != '.csv':
                    print(f"  Converted {file_path.name} -> train/{output_csv.name}")
    
    # Handle test files alone (unusual but possible)
    if test_files and not train_files:
        for file_path in test_files:
            output_csv = test_dir / f"{file_path.stem}.csv"
            if convert_to_csv(file_path, output_csv):
                if file_path.suffix.lower() != '.csv':
                    print(f"  Converted {file_path.name} -> test/{output_csv.name}")


def process_root_csv_files(real_data_dir: Path) -> None:
    """Process CSV files directly in the root of real_data."""
    root_csvs = [f for f in real_data_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() == '.csv']
    
    for csv_file in root_csvs:
        dataset_name = get_dataset_name_from_file(csv_file)
        dataset_dir = real_data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        train_dir = dataset_dir / "train"
        train_dir.mkdir(exist_ok=True)
        
        # Move CSV to train folder
        output_csv = train_dir / csv_file.name
        if not output_csv.exists():
            shutil.move(str(csv_file), str(output_csv))
            print(f"  Moved {csv_file.name} -> {dataset_name}/train/{output_csv.name}")
        else:
            print(f"  Skipped {csv_file.name} (already exists in {dataset_name}/train/)")


def process_root_non_csv_files(real_data_dir: Path) -> None:
    """Process non-CSV files directly in the root of real_data."""
    root_files = [f for f in real_data_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in ['.arff', '.data', '.xlsx', '.xls', '.nna']]
    
    for file_path in root_files:
        dataset_name = get_dataset_name_from_file(file_path)
        dataset_dir = real_data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        train_dir = dataset_dir / "train"
        train_dir.mkdir(exist_ok=True)
        
        # Convert to CSV
        output_csv = train_dir / f"{file_path.stem}.csv"
        if convert_to_csv(file_path, output_csv):
            print(f"  Converted {file_path.name} -> {dataset_name}/train/{output_csv.name}")
            # Remove original if conversion successful
            file_path.unlink()


def process_existing_folders(real_data_dir: Path) -> None:
    """Process existing dataset folders."""
    dataset_dirs = [d for d in real_data_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in dataset_dirs:
        # Skip if already has train/test structure with CSVs
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"
        
        if train_dir.exists() and any(train_dir.glob("*.csv")):
            print(f"Skipping {dataset_dir.name} (already organized)")
            continue
        
        print(f"\nProcessing folder: {dataset_dir.name}")
        organize_dataset_folder(dataset_dir)


def main():
    """Main function to organize all datasets."""
    real_data_dir = Path(REAL_DATA_DIR)
    
    if not real_data_dir.exists():
        print(f"Error: {real_data_dir} does not exist!")
        return
    
    print("=" * 60)
    print("Organizing datasets in real_data folder")
    print("=" * 60)
    
    # Step 1: Process CSV files in root
    print("\n[Step 1] Processing CSV files in root...")
    process_root_csv_files(real_data_dir)
    
    # Step 2: Process non-CSV files in root
    print("\n[Step 2] Processing non-CSV files in root...")
    process_root_non_csv_files(real_data_dir)
    
    # Step 3: Process existing folders
    print("\n[Step 3] Processing existing dataset folders...")
    process_existing_folders(real_data_dir)
    
    print("\n" + "=" * 60)
    print("Organization complete!")
    print("=" * 60)
    print("\nFinal structure should be:")
    print("  data/real_data/")
    print("    <dataset_name>/")
    print("      train/")
    print("        *.csv")
    print("      test/")
    print("        *.csv (if available)")


if __name__ == "__main__":
    main()

