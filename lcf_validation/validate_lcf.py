# %% Block 1: Single Dataset Loading for LCF Validation
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
lcf_data_dir = "LCF Data"  # Directory containing Excel workbooks
sheet_name = "Treppe Hilfe"  # Sheet name to extract from each workbook

# %% Load single Excel workbook for testing
def load_single_dataset(file_path, sheet_name="Treppe Hilfe"):
    """
    Load and prepare a single LCF dataset from Excel workbook
    
    Args:
        file_path: Path to Excel file
        sheet_name: Name of sheet containing data
    
    Returns:
        DataFrame with columns: load, cycles, fracture (PyLife format)
    """
    try:
        # Read Excel sheet
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=5)
        
        # Clean column names (remove any whitespace/special chars)
        df_raw.columns = df_raw.columns.str.strip()
        
        # Extract load and cycles columns
        load_col = 'Fo'  # Load/stress values
        cycles_col = 'Zyklen'  # Cycles to failure
        
        # Create PyLife format DataFrame
        df_pylife = pd.DataFrame({
            'load': df_raw[load_col],
            'cycles': df_raw[cycles_col],
            'fracture': True,  # All LCF data points are failures
            'censor': 1
        })
        
        # Filter out invalid rows (zero values, NaN)
        df_pylife = df_pylife[
            (df_pylife['load'] > 0) & 
            (df_pylife['cycles'] > 0) & 
            (df_pylife['load'].notna()) & 
            (df_pylife['cycles'].notna())
        ].reset_index(drop=True)
        
        return df_pylife
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

# %% Test with specific file
test_file = Path(lcf_data_dir) / "5_Protokoll_EBlech_Schaeffler1_gekerbtR1_Voestalpine_MF.xlsx"
print(f"Testi ng with file: {test_file.name}")

# Load and print the dataframe
df_test = load_single_dataset(test_file, sheet_name)
print("\nDataFrame:")
print(df_test)
# %%
