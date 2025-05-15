# %% [markdown]
# Wöhler analysis of fatigue data using pylife

# %% [markdown]
# **Import Statements:**

# %%
# import statements 

import os
import traceback
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import norm, linregress
import pylife.materialdata.woehler as woehler
from pylife.materiallaws import WoehlerCurve
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime


# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %%
# Constants
N_LCF = 10000  # Pivot point in LCF
NG = 4000000  # Maximum number of cycles

# set analyzer to "pylife" or "hybrid"
analyzer = "pylife"

# %% analyze_fatigue_pylife

def analyze_fatigue_pylife(file_path):
    """
    Analyze a single fatigue test file using pylife MaxLikeInf method
    
    Parameters:
    file_path : str, path to Excel file containing test data and reference results
    
    Returns:
    dict with calculated and reference results
    """
    try:
        print("\n=== Starting Analysis ===")
        print(f"Processing file: {file_path}")
        
        # Read test data
        df_test = pd.read_excel(file_path, sheet_name='Data')
        print("\nInput Data:")
        print(f"Columns found: {df_test.columns.tolist()}")
        
        # Rename 'loads' to 'load' if necessary
        if 'loads' in df_test.columns:
            df_test = df_test.rename(columns={'loads': 'load'})
            
        # Read reference values    
        df_ref = pd.read_excel(file_path, sheet_name='Jurojin_results', header=None)
        
        # Extract reference values
        ref_values = {}
        for idx, row in df_ref.iterrows():
            param_name = row[0]
            param_value = row[1]
            if isinstance(param_value, str) and ',' in param_value:
                param_value = float(param_value.replace(',', '.'))
            ref_values[param_name] = param_value
            
        print("\nReference values:")
        for key, value in ref_values.items():
            print(f"{key}: {value}")
        
        # Prepare data and run analysis
        print("\nPreparing data for analysis...")
        df_prepared = df_test[['load', 'cycles', 'censor']]
        results = analyze_with_maxlikeinf(df_prepared, NG, ref_values)
        
        return results
        
    except Exception as e:
        print(f"\nError processing {file_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        return None

def analyze_with_maxlikeinf(df, NG, ref_values):
    """
    Analyze using only MaxLikeInf method
    """
    print("\n=== MaxLikeInf Analysis ===")
    
    # Prepare data for pylife
    print("Determining fractures...")
    df = woehler.determine_fractures(df, NG)
    fatigue_data = df.fatigue_data
    
    print("\nRunning MaxLikeInf analysis...")
    result = woehler.MaxLikeInf(fatigue_data).analyze()
    
    # Calculate parameters
    calc_k = round(result.k_1, 2)
    calc_ND = int(round(result.ND, 0))
    calc_Pu50 = int(round(result.SD, 0))
    calc_slog = round(np.log10(result.TS)/2.5361, 3)
    
    print("\nCalculated parameters:")
    print(f"k: {calc_k}")
    print(f"ND: {calc_ND}")
    print(f"Pü50: {calc_Pu50}")
    print(f"slog: {calc_slog}")
    
    # Calculate differences
    results = {
        'calc_k': calc_k,
        'calc_ND': calc_ND,
        'calc_Pu50': calc_Pu50,
        'calc_slog': calc_slog,
        'ref_k': ref_values['k'],
        'ref_ND': int(ref_values['ND']),
        'ref_Pu50': ref_values['Pü50'],
        'ref_slog': ref_values['slog'],
        'diff_k_pct': (calc_k - ref_values['k'])/ref_values['k'] * 100,
        'diff_ND_pct': (calc_ND - ref_values['ND'])/ref_values['ND'] * 100,
        'diff_Pu50_pct': (calc_Pu50 - ref_values['Pü50'])/ref_values['Pü50'] * 100,
        'diff_slog_pct': (calc_slog - ref_values['slog'])/ref_values['slog'] * 100
    }
    
    print("\nPercentage differences from reference:")
    print(f"k: {results['diff_k_pct']:.2f}%")
    print(f"ND: {results['diff_ND_pct']:.2f}%")
    print(f"Pü50: {results['diff_Pu50_pct']:.2f}%")
    print(f"slog: {results['diff_slog_pct']:.2f}%")
    
    return results


# %% hybrid maxlike class definition
class HybridMaxLikeInf(woehler.MaxLikeInf):
    def __init__(self, fatigue_data, file_path=None):
        super().__init__(fatigue_data)
        self.file_path = file_path
        
        # Calculate N_LCF from data
        all_cycles = np.concatenate([
            self._fd.fractures.cycles, 
            self._fd.runouts.cycles
        ])
        self.N_LCF = int(np.min(all_cycles))  # Get lowest cycle value
        print(f"Setting N_LCF to lowest cycle value: {self.N_LCF}")
    
    def create_sn_plot(self, df, SD, k1, ND, TS, N_LCF, NG):
        """Create SN curve plot using calculated parameters"""
        # Use self.N_LCF instead of hardcoded value
        N_LCF = self.N_LCF
        
        # Create figure
        fig = make_subplots()
        
        # Separate failures and survivors
        failures = df[df['censor'] == 1]
        survivors = df[df['censor'] == 0]
        
        # Plot data points
        if not failures.empty:
            fig.add_trace(go.Scatter(
                x=failures['cycles'], y=failures['load'],
                mode='markers', marker=dict(color='#648fff', symbol='cross'),
                name='Failures',
                hovertemplate='Cycles: %{x:.1f}<br>Load: %{y}<br>Status: Failure<extra></extra>'
            ))
        
        if not survivors.empty:
            fig.add_trace(go.Scatter(
                x=survivors['cycles'], y=survivors['load'],
                mode='markers', marker=dict(color='#648fff', symbol='triangle-right'),
                name='Survivors',
                hovertemplate='Cycles: %{x:.1f}<br>Load: %{y}<br>Status: Survivor<extra></extra>'
            ))
        
        # Calculate and plot LCF curve
        L_LCF = 10**(np.log10(SD)-(np.log10(ND/N_LCF))/-k1)
        x_LCF = [N_LCF, ND]
        y_LCF = [L_LCF, SD]
        
        fig.add_trace(go.Scatter(
            x=x_LCF, y=y_LCF,
            mode='lines', line=dict(color='#648fff'),
            name='LCF'
        ))
        
        # Plot HCF curve
        x_HCF = [ND, NG]
        y_HCF = [SD, SD]
        fig.add_trace(go.Scatter(
            x=x_HCF, y=y_HCF,
            mode='lines', line=dict(color='#648fff', dash='dash'),
            name='HCF'
        ))
        
        # Calculate slog
        slog = np.log10(TS)/2.5361
        
        # Add parameter text
        param_text = (
            f"<b>{os.path.basename(self.file_path)}</b><br>"
            f"k = {k1:.2f}, "
            f"ND = {int(ND):,}, "
            f"Pü50 = {SD:.1f}, "
            f"slog = {slog:.3f}, "
            f"N_LCF = {N_LCF:,}, "
            f"NG = {NG:,}"
        )
        
        # Update layout
        fig.update_layout(
            title=f'SN Curve<br><sub>{param_text}</sub>',
            xaxis_type="log",
            yaxis_type="log",
            xaxis_title="Cycles",
            yaxis_title="Load",
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def _specific_analysis(self, wc):
        optimization_steps = []
        
        print("\n=== HybridMaxLikeInf Analysis ===")
        
        # Calculate initial parameters
        slope, lg_intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(self._fd.fractures.load),
            np.log10(self._fd.fractures.cycles)
        )
        
        # Initialize optimization
        SD_start = self._fd.fatigue_limit
        TS_start = 1.2
        
        def debug_objective(p):
            likelihood = self._lh.likelihood_infinite(p[0], p[1])
            optimization_steps.append({
                'Step': len(optimization_steps) + 1,
                'SD': p[0],
                'TS': p[1],
                'Likelihood': likelihood,
                'Method': 'Nelder-Mead'
            })
            return -likelihood
        
        # Try Nelder-Mead first
        print("Starting with Nelder-Mead optimization...")
        nm_result = optimize.fmin(debug_objective, [SD_start, TS_start], 
                                disp=True, full_output=True)
        
        # Extract results from Nelder-Mead
        SD = nm_result[0][0]
        TS = nm_result[0][1]
        
        # Define thresholds for "wonky" values
        SD_MIN_THRESHOLD = 50  # Minimum reasonable fatigue strength
        TS_MAX_THRESHOLD = 2.0 # Maximum reasonable scatter
        
        # If values are extreme, switch to L-BFGS-B with bounds
        if SD < SD_MIN_THRESHOLD or TS > TS_MAX_THRESHOLD:
            print("\nExtreme values detected! Switching to bounded optimization...")
            print(f"Current SD: {SD}, Current TS: {TS}")
            
            def bounded_objective(p):
                likelihood = self._lh.likelihood_infinite(p[0], p[1])
                optimization_steps.append({
                    'Step': len(optimization_steps) + 1,
                    'SD': p[0],
                    'TS': p[1],
                    'Likelihood': likelihood,
                    'Method': 'L-BFGS-B'
                })
                return -likelihood
            
            # Define bounds
            bounds = [(SD_MIN_THRESHOLD, 30000),  # SD bounds
                     (1.0, TS_MAX_THRESHOLD)]    # TS bounds
            
            # Use L-BFGS-B with bounds
            lbfgs_result = optimize.minimize(bounded_objective, 
                                          [SD_start, TS_start],
                                          method='L-BFGS-B',
                                          bounds=bounds)
            
            # Extract results from L-BFGS-B
            SD = lbfgs_result.x[0]
            TS = lbfgs_result.x[1]
            print(f"\nBounded optimization results:")
            print(f"Final SD: {SD}")
            print(f"Final TS: {TS}")
        
        # Create plots
        df_steps = pd.DataFrame(optimization_steps)
        
        # Convergence plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['Likelihood'], 
                                mode='lines+markers', name='Likelihood'))
        fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['SD'], 
                                mode='lines+markers', name='SD'))
        fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['TS'], 
                                mode='lines+markers', name='TS'))
        
        fig.update_layout(
            title='Convergence Plot',
            xaxis_title='Step',
            yaxis_title='Value',
            width=800,
            height=600
        )
        
        fig.show()

        # Export to Excel with optimization method information
        filename = f'optimization_path_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        with pd.ExcelWriter(filename) as writer:
            df_steps.to_excel(writer, sheet_name='Optimization_Path', index=False)
            pd.DataFrame([{
                'R_squared': r_value**2,
                'Initial_SD': SD_start,
                'Initial_TS': TS_start,
                'Final_SD': SD,
                'Final_TS': TS,
                'Final_Likelihood': optimization_steps[-1]['Likelihood'],
                'Number_of_Steps': len(optimization_steps),
                'Used_Bounded_Optimization': 'Yes' if len(df_steps['Method'].unique()) > 1 else 'No'
            }]).to_excel(writer, sheet_name='Summary', index=False)
        
        # Calculate ND and return results
        ND = 10**(lg_intercept + slope * (np.log10(SD)))
        
        plot_df = pd.DataFrame({
        'cycles': np.concatenate([self._fd.fractures.cycles, self._fd.runouts.cycles]),
        'load': np.concatenate([self._fd.fractures.load, self._fd.runouts.load]),
        'censor': np.concatenate([np.ones(len(self._fd.fractures)), np.zeros(len(self._fd.runouts))])
        })
        
        sn_plot = self.create_sn_plot(plot_df, SD, -slope, ND, TS, N_LCF, NG)
        sn_plot.show()
        
        wc['SD'] = SD
        wc['TS'] = TS
        wc['ND'] = ND
        wc['k_1'] = -slope
        
        return wc

# %% analyze_fatigue_hybrid
def analyze_fatigue_hybrid(file_path):
    """
    Analyze a single fatigue test file using hybrid MaxLikeInf method
    """
    try:
        print("\n=== Starting Analysis ===")
        print(f"Processing file: {file_path}")
        
        # Read test data
        df_test = pd.read_excel(file_path, sheet_name='Data')
        print("\nInput Data:")
        print(f"Columns found: {df_test.columns.tolist()}")
        
        # Rename 'loads' to 'load' if necessary
        if 'loads' in df_test.columns:
            df_test = df_test.rename(columns={'loads': 'load'})
            
        # Read reference values    
        df_ref = pd.read_excel(file_path, sheet_name='Jurojin_results', header=None)
        
        # Extract reference values
        ref_values = {}
        for idx, row in df_ref.iterrows():
            param_name = row[0]
            param_value = row[1]
            if isinstance(param_value, str) and ',' in param_value:
                param_value = float(param_value.replace(',', '.'))
            ref_values[param_name] = param_value
        
        # Prepare data and run analysis
        print("\nPreparing data for analysis...")
        df_prepared = df_test[['load', 'cycles', 'censor']]
        df_prepared = woehler.determine_fractures(df_prepared, NG)
        fatigue_data = df_prepared.fatigue_data
        
        # Run hybrid analysis
        result = HybridMaxLikeInf(fatigue_data, file_path).analyze()
        
        # Inspect fatigue_data: class FatigueData from pylife
        
        # print(f"Type of fatigue_data is ", type(fatigue_data)) 
        # print("Fractures:")
        # print(fatigue_data.fractures)
        # print("\nRunouts:")
        # print(fatigue_data.runouts)
        
        # Calculate parameters
        calc_k = round(result.k_1, 2)
        calc_ND = int(round(result.ND, 0))
        calc_Pu50 = int(round(result.SD, 0))
        calc_slog = round(np.log10(result.TS)/2.5361, 3)
        
        print("\nReference parameters:")
        for key, value in ref_values.items():
            print(f"{key}: {value}")
        
        print("\nCalculated parameters:")
        print(f"k: {calc_k}")
        print(f"ND: {calc_ND}")
        print(f"Pü50: {calc_Pu50}")
        print(f"slog: {calc_slog}")
        
        # Calculate differences
        results = {
            'calc_k': calc_k,
            'calc_ND': calc_ND,
            'calc_Pu50': calc_Pu50,
            'calc_slog': calc_slog,
            'ref_k': ref_values['k'],
            'ref_ND': int(ref_values['ND']),
            'ref_Pu50': ref_values['Pü50'],
            'ref_slog': ref_values['slog'],
            'diff_k_pct': (calc_k - ref_values['k'])/ref_values['k'] * 100,
            'diff_ND_pct': (calc_ND - ref_values['ND'])/ref_values['ND'] * 100,
            'diff_Pu50_pct': (calc_Pu50 - ref_values['Pü50'])/ref_values['Pü50'] * 100,
            'diff_slog_pct': (calc_slog - ref_values['slog'])/ref_values['slog'] * 100
        }
        
        print("\nPercentage differences from reference:")
        print(f"k: {results['diff_k_pct']:.2f}%")
        print(f"ND: {results['diff_ND_pct']:.2f}%")
        print(f"Pü50: {results['diff_Pu50_pct']:.2f}%")
        print(f"slog: {results['diff_slog_pct']:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"\nError processing {file_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        return None

# %% [markdown]
# MaxLikeInf High deviation (>5%)
# 
# 240221_1.xlsx
# 241028.xlsx
# 242039G_Evaluation.xlsx
# 4PB_12.xlsx
# 4PB_14.xlsx
# 4PB_15.xlsx
# 4PB_2.xlsx
# 4PB_6.xlsx
# LH_intake.xlsx
# NO027_ungekerbt.xlsx
# NO35_gekerbt.xlsx
# NO35_ungekerbt.xlsx
# Scorpion RE_long_Arm.xlsx
# Scorpion_Li_long.xlsx
# 
# 

# %%

# Test with single file
file_path = "Evaluation data.xlsx"
print("\n=== Starting Analysis of Single Dataset ===")
print(f"Loading file: {file_path}")

# Load, display input data
df = pd.read_excel(file_path, sheet_name='Data')
print("\nInput Data Summary:")
print(f"Number of total points: {len(df)}")
print(f"Number of failures: {len(df[df['censor'] == 1])}")
print(f"Number of survivors: {len(df[df['censor'] == 0])}")
print("\nFirst few rows of data:")
print(df.head())

if analyzer == "pylife":
    results = analyze_fatigue_pylife(file_path)
elif analyzer == "hybrid":
    results = analyze_fatigue_hybrid(file_path)

# %%
def batch_analyze_files(folder_path, output_filename=None):
    """
    Analyze all Excel files in the specified folder using HybridMaxLikeInf
    """
    # Step 1: Set up default output filename with timestamp
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'fatigue_analysis_{os.path.basename(folder_path)}_{timestamp}.xlsx'
    
    # Step 2: Initialize tracking variables (unchanged)
    all_results = []
    processed_files = 0
    failed_files = []
    
    print(f"\nProcessing all Excel files in: {folder_path}")
    
    # Step 3: Process each file (mostly unchanged)
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):
            try:
                file_path = os.path.join(folder_path, filename)
                print(f"\nProcessing file {processed_files + 1}: {filename}")
                
                # Step 4: Use either analyze_fatigue_pylife or analyze_fatigue_file function
                
                if analyzer == "pylife":
                    results = analyze_fatigue_pylife(file_path)
                elif analyzer == "hybrid":
                    results = analyze_fatigue_hybrid(file_path)
                
                # Step 5: Add filename to results dictionary for reference
                if results is not None:
                    results['filename'] = filename
                    all_results.append(results)
                    processed_files += 1
                else:
                    failed_files.append(filename)
                    
            except Exception as e:
                print(f"Error processing {filename}:")
                print(str(e))
                failed_files.append(filename)
    
    # Step 6: Create and save results summary
    if all_results:
        # Create DataFrame from results
        final_results = pd.DataFrame(all_results)
        
        # Step 7: Organize columns in a meaningful order
        column_order = [
            'filename',
            'calc_k', 'ref_k', 'diff_k_pct',
            'calc_ND', 'ref_ND', 'diff_ND_pct',
            'calc_Pu50', 'ref_Pu50', 'diff_Pu50_pct',
            'calc_slog', 'ref_slog', 'diff_slog_pct'
        ]
        final_results = final_results[column_order]
        
        try:
            # Step 8: Save to Excel with some formatting
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                final_results.to_excel(writer, sheet_name='Results', index=False)
                
                # Optional: Add a summary sheet
                summary = pd.DataFrame({
                    'Metric': ['Files Processed', 'Files Failed'],
                    'Value': [processed_files, len(failed_files)]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                if failed_files:
                    pd.DataFrame({'Failed Files': failed_files}).to_excel(
                        writer, sheet_name='Failed Files', index=False)
            
            print(f"\nResults successfully saved to {output_filename}")
            print(f"\nProcessed {processed_files} files successfully")
            if failed_files:
                print(f"Failed to process {len(failed_files)} files:")
                for file in failed_files:
                    print(f"- {file}")
            
            return final_results
            
        except Exception as e:
            print(f"Error saving results to Excel: {str(e)}")
            return final_results
    else:
        print("No results were successfully processed!")
        return None

# %%
# Run script
if __name__ == "__main__":
        # Constants
    N_LCF = 10000  # Pivot point in LCF
    NG = 5000000  # Maximum number of cycles
    folder_path = "4PB"  # or whatever your folder name is
    results = batch_analyze_files(folder_path)

# %%
# Run script
if __name__ == "__main__":
        # Constants
    N_LCF = 10000  # Pivot point in LCF
    NG = 10000000  # Maximum number of cycles
    folder_path = "NO"  # or whatever your folder name is
    results = batch_analyze_files(folder_path)

# %%
# Run script
if __name__ == "__main__":
        # Constants
    N_LCF = 10000  # Pivot point in LCF
    NG = 4000000  # Maximum number of cycles
    folder_path = "Troy"  # or whatever your folder name is
    results = batch_analyze_files(folder_path)

# %%

# Test with single file
file_path = "Troy\\LH_intake.xlsx"
print("\n=== Starting Analysis of Single Dataset ===")
print(f"Loading file: {file_path}")

# Load, display input data
df = pd.read_excel(file_path, sheet_name='Data')
print("\nInput Data Summary:")
print(f"Number of total points: {len(df)}")
print(f"Number of failures: {len(df[df['censor'] == 1])}")
print(f"Number of survivors: {len(df[df['censor'] == 0])}")
print("\nFirst few rows of data:")
print(df.head())

if analyzer == "pylife":
    results = analyze_fatigue_pylife(file_path)
elif analyzer == "hybrid":
    results = analyze_fatigue_hybrid(file_path)