# %% Imports
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pylife.materialdata.woehler as woehler
from datetime import datetime

# %% Configuration
lcf_data_dir = "LCF Data"  # Directory containing Excel workbooks
sheet_name = "Treppe Hilfe"  # Sheet name to extract from each workbook

# %% Block 1: Single Dataset Loading for LCF Validation
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


# %% Block 2: Elementary Analysis Function

def run_elementary_analysis(df_pylife, dataset_name):
    """
    Run PyLife Elementary analysis on loaded dataframe
    
    Args:
        df_pylife: DataFrame with columns [load, cycles, censor]
        dataset_name: Name of the dataset for identification
    
    Returns:
        Dictionary with results
    """
    try:
        # Convert to PyLife FatigueData format
        # Note: PyLife expects 'fracture' column (True=failure), not 'censor'
        df_analysis = df_pylife.copy()
        df_analysis['fracture'] = df_analysis['censor'] == 1  # Convert censor to fracture
        
        # Create FatigueData object
        fatigue_data = df_analysis.fatigue_data
        
        # Run Elementary analysis
        analyzer = woehler.Elementary(fatigue_data)
        result = analyzer.analyze()
        
        # Extract k_1 and ND values
        results_dict = {
            'dataset_name': dataset_name,
            'k_1': result.k_1,
            'ND': result.ND,
            'SD': result.SD,
            'TN': result.TN,
            'TS': result.TS,
            'status': 'Success',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        return results_dict
        
    except Exception as e:
        return {
            'dataset_name': dataset_name,
            'k_1': None,
            'ND': None,
            'SD': None,
            'TN': None,
            'TS': None,
            'status': f'Error: {str(e)}',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }


# %% Block 3: Save Results Function

def save_results_to_excel(results_dict, output_file="LCF_Validation_Results.xlsx"):
    """
    Save results to Excel, updating existing dataset or adding new row
    
    Args:
        results_dict: Dictionary with analysis results
        output_file: Excel file path
    """
    try:
        # Check if file exists
        if Path(output_file).exists():
            # Load existing data
            df_existing = pd.read_excel(output_file)
            
            # Check if dataset already exists
            dataset_mask = df_existing['dataset_name'] == results_dict['dataset_name']
            
            if dataset_mask.any():
                # Update existing row
                for col in results_dict.keys():
                    if col in df_existing.columns:
                        df_existing.loc[dataset_mask, col] = results_dict[col]
                print(f"Updated existing dataset: {results_dict['dataset_name']}")
            else:
                # Add new row
                new_row = pd.DataFrame([results_dict])
                df_existing = pd.concat([df_existing, new_row], ignore_index=True)
                print(f"Added new dataset: {results_dict['dataset_name']}")
            
            df_final = df_existing
        else:
            # Create new file
            df_final = pd.DataFrame([results_dict])
            print(f"Created new file with dataset: {results_dict['dataset_name']}")
        
        # Save to Excel
        df_final.to_excel(output_file, index=False)
        print(f"Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False


# %% Block 4: Batch Processing Function
def process_lcf_directory(data_directory="LCF Data", sheet_name="Treppe Hilfe", output_file="LCF_Validation_Results.xlsx"):
    """
    Batch process all Excel files in LCF directory
    
    Args:
        data_directory: Directory containing Excel workbooks
        sheet_name: Sheet name to extract from each workbook
        output_file: Excel file to save results
    """
    print(f"=== Batch Processing LCF Validation from '{data_directory}' ===")
    
    # Find all Excel files
    if not Path(data_directory).exists():
        print(f"❌ Directory '{data_directory}' not found!")
        return
    
    excel_files = list(Path(data_directory).glob("*.xlsx")) + list(Path(data_directory).glob("*.xls"))
    excel_files = [f for f in excel_files if not f.name.startswith('~$')]  # Exclude temp files
    
    print(f"Found {len(excel_files)} Excel files")
    
    successful_count = 0
    failed_count = 0
    
    for file_path in excel_files:
        print(f"\n--- Processing {file_path.name} ---")
        
        try:
            # Extract dataset name (remove extension)
            dataset_name = file_path.stem
            
            # Load dataset
            df_data = load_single_dataset(file_path, sheet_name)
            
            if df_data is not None and len(df_data) > 0:
                # Run analysis
                results = run_elementary_analysis(df_data, dataset_name)
                
                # Save results
                if save_results_to_excel(results, output_file):
                    successful_count += 1
                    print(f"✅ {dataset_name}: k_1={results.get('k_1', 'N/A')}, ND={results.get('ND', 'N/A')}")
                else:
                    failed_count += 1
                    print(f"❌ Failed to save results for {dataset_name}")
            else:
                failed_count += 1
                print(f"❌ Failed to load data from {file_path.name}")
                
        except Exception as e:
            failed_count += 1
            print(f"❌ Error processing {file_path.name}: {str(e)}")
    
    print(f"\n=== Batch Processing Complete ===")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"Results saved to: {output_file}")


# %% Test single data loading with specific file
dataset_name = "5_Protokoll_EBlech_Schaeffler1_gekerbtR1_Voestalpine_MF"
test_file = Path(lcf_data_dir) / f"{dataset_name}.xlsx"
print(f"Testing with file: {test_file.name}")

# Load and print the dataframe
df_test = load_single_dataset(test_file, sheet_name)
print("\nDataFrame:")
print(df_test)

# %% Test Elementary Analysis
results = run_elementary_analysis(df_test, dataset_name)
print("Elementary Analysis Results:")
print(results)

# %% Test Save Function
save_results_to_excel(results, "LCF_Validation_Results.xlsx")


# %% Run Batch Processing
process_lcf_directory("LCF Data", "Treppe Hilfe", "LCF_Validation_Results.xlsx")
# %%
