# %% [markdown] 
# # Woehler Analysis Validation

import os

import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import norm, linregress
import pylife.materialdata.woehler as woehler
from pylife.materiallaws import WoehlerCurve
from pylife.materialdata.woehler.likelihood import Likelihood

from scipy import optimize
from scipy import stats
from datetime import datetime


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %% display float values to 2 decimal places

pd.options.display.float_format = "{:.2f}".format

# %% Prepare data for analysis

from woehler_utils import *

# Sheet containing data must be named 'Data'

file_path = "All Data/4PB_11.xlsx"

N_LCF = 10000  # Pivot point in LCF
NG = 5000000   # Maximum number of cycles
switch = False # Change to True to run L-BFGS-B

df_test, ref_values = analyze_fatigue_file(file_path)
df_prepared = df_test[['load', 'cycles', 'censor']]
df_prepared = woehler.determine_fractures(df_prepared, NG)
fatigue_data = df_prepared.fatigue_data


# %% CONTROL: Run normal MaxLikeInf analysis BEFORE running further blocks
analyzer = woehler.MaxLikeInf(fatigue_data)
result = analyzer.analyze()
print(f"Pylife results out of the box:")
print(f"SD: {result.SD:.2f}")
print(f"TS: {result.TS:.2f}")
print(f"ND: {result.ND:.2f}")
print(f"k_1: {result.k_1:.2f}")
print(f"slog: {np.log10(result.TS)/2.5361:.2f}")

# result_df = result.to_frame().round(2)
# result_df


# %% Track optimization progress with MaxLikeInf

# Create list to store optimization steps
optimization_steps = []

# Create likelihood object
lh = Likelihood(fatigue_data)

# Run Nelder-Mead first
nm_results = run_optimization_with_tracking(lh, [fatigue_data.fatigue_limit, 1.2], method='nelder-mead')

# Try L-BFGS-B if needed
if not nm_results['success'] or not nm_results['reasonable_values'] or switch==True:
    print("\nNelder-Mead failed or produced unreasonable values. Trying L-BFGS-B...")
    bounds = [(fatigue_data.load.min() * 0.5, fatigue_data.load.max() * 2.0), (1.0, 10.0)]
    
    lbfgs_results = run_optimization_with_tracking(
        lh, 
        [fatigue_data.fatigue_limit, 1.2], 
        method='l-bfgs-b',
        bounds=bounds
    )
    
    # Compare results
    print("\nComparison of methods:")
    print(f"Nelder-Mead: SD={nm_results['SD']:.2f}, TS={nm_results['TS']:.2f}, Success={nm_results['success']}")
    print(f"L-BFGS-B: SD={lbfgs_results['SD']:.2f}, TS={lbfgs_results['TS']:.2f}, Success={lbfgs_results['success']}")


# %% Run Huck method analysis
from huck import HuckMethod

def analyze_with_huck(fatigue_data):
    """Run analysis using Huck's method (staircase-based)"""
    print("\n=== Running Huck's Method Analysis (Staircase Region Only) ===")
    
    # Create analyzer and run analysis
    analyzer = HuckMethod(fatigue_data)
    result = analyzer.analyze()
    
    # Calculate slog from TS
    slog = np.log10(result.TS)/2.5361
    
    # Print summary
    print("\nHuck's Method Results:")
    print(f"SD (Pü50): {result.SD:.2f}  <- Primary result from Huck's method")
    print(f"TS: {result.TS:.4f}  <- Derived from staircase region only")
    print(f"slog: {slog:.4f}  <- Derived from staircase region only")
    print(f"ND: {result.ND:.2f}  <- Inherited from Elementary analysis")
    print(f"k_1: {result.k_1:.2f}  <- Inherited from Elementary analysis")
    
    # Visualize staircase
    analyzer.plot_staircase()
    
    return result

# %% Compare MaxLikeInf to Huck method 

#!TODO move to woehler_utils.py
def compare_methods(fatigue_data, ref_values=None, dataset_name="Unknown"):
    """Compare MaxLikeInf (Nelder-Mead), L-BFGS-B and Huck method results"""
    results_list = []  # Will store all results as rows
    
    elementary_analyzer = woehler.Elementary(fatigue_data)
    elementary_result = elementary_analyzer.analyze()
    print(f"Elementary ND (staircase boundary): {elementary_result.ND:.0f} cycles")
        
    # Run Nelder-Mead
    ml_analyzer = woehler.MaxLikeInf(fatigue_data)
    ml_result = ml_analyzer.analyze()
    ml_slog = np.log10(ml_result.TS)/2.5361
    
    # Create row for Nelder-Mead results
    nm_row = {
        'Dataset': dataset_name,
        'Method': 'Nelder-Mead',
        'SD (Pü50)': ml_result.SD,
        'TS': ml_result.TS,
        'slog': ml_slog,
        'ND': ml_result.ND,
        # 'k': ml_result.k_1,
        'Optimizer Message': 'Standard analysis',
        'Convergence Plot': 'N/A'
    }
    results_list.append(nm_row)
    
    # Check if Nelder-Mead results are reasonable
    min_load = fatigue_data.load.min()
    max_load = fatigue_data.load.max()
    nm_reasonable = True
    
    if ml_result.SD < min_load * 0.5 or ml_result.SD > max_load * 2.0 or ml_result.TS < 1.0 or ml_result.TS > 50.0:
        nm_reasonable = False
        print("Nelder-Mead results appear unreasonable. Running L-BFGS-B...")
        
        # Update Nelder-Mead row with warning
        nm_row['Optimizer Message'] = 'Unreasonable values detected'
        
        # Run L-BFGS-B
        lh = Likelihood(fatigue_data)
        bounds = [(min_load * 0.5, max_load * 2.0), (1.0, 10.0)]
        lbfgs_results = run_optimization_with_tracking(
            lh, 
            [fatigue_data.fatigue_limit, 1.2], 
            method='l-bfgs-b',
            bounds=bounds
        )

        # Create L-BFGS-B results row
        lbfgs_sd, lbfgs_ts = lbfgs_results['SD'], lbfgs_results['TS']
        lbfgs_slog = np.log10(lbfgs_ts)/2.5361

        # Recalculate ND using L-BFGS-B optimized SD (consistent with MaxLikeInf)
        lbfgs_nd = ml_analyzer._transition_cycles(lbfgs_sd)
        
        lbfgs_row = {
            'Dataset': dataset_name,
            'Method': 'L-BFGS-B',
            'SD (Pü50)': lbfgs_sd,
            'TS': lbfgs_ts,
            'slog': lbfgs_slog,
            'ND': lbfgs_nd,
            'Optimizer Message': lbfgs_results['message'],
            'Convergence Plot': 'See plot in notebook'
        }
        results_list.append(lbfgs_row)
        
        # Create a Series for return value
        lbfgs_series = pd.Series({
            'SD': lbfgs_sd,
            'TS': lbfgs_ts,
            'ND': lbfgs_nd,
        })
    else:
        lbfgs_series = None    
    
    # Run Huck method
    huck_analyzer = HuckMethod(fatigue_data)
    huck_result = huck_analyzer.analyze()
    huck_slog = np.log10(huck_result.TS)/2.5361
    
    # Create Huck method row
    huck_row = {
        'Dataset': dataset_name,
        'Method': 'Huck (Staircase)',
        'SD (Pü50)': huck_result.SD,
        'TS': huck_result.TS,
        'slog': huck_slog,
        'ND': huck_result.ND,
        # 'k': huck_result.k_1,
        'Optimizer Message': 'SD from staircase region; ND and k from Elementary',
        'Convergence Plot': 'N/A'
    }
    results_list.append(huck_row)
    
    # Add reference values if available
    if ref_values is not None:
        ref_row = {
            'Dataset': dataset_name,
            'Method': 'Jurojin Reference',
            'SD (Pü50)': ref_values.get('Pü50', None),
            'TS': ref_values.get('TS', None),
            'slog': ref_values.get('slog', None),
            'ND': ref_values.get('ND', None),
            'k': ref_values.get('k', None),
            'Optimizer Message': 'N/A',
            'Convergence Plot': 'N/A'
        }
        results_list.append(ref_row)
    
    # Create DataFrame from list of results
    results_df = pd.DataFrame(results_list)
    
    # Display comparison table
    print("\nComparison of Methods:")
    display_df = results_df[['Method', 'SD (Pü50)', 'TS', 'slog', 'ND']]
    print(display_df)
    
    # Create and display comparison plot
    df_prepared = fatigue_data._obj
    plot = create_comparison_plot(
        df_prepared, 
        elementary_result, 
        ml_result, 
        huck_result,
        file_name=dataset_name
    )
    plot.show()
    
    # Return complete results DataFrame along with original objects
    return results_df, ml_result, lbfgs_series if not nm_reasonable else None, huck_result


def save_comparison_to_excel(results_df, filename=None):
    """Save comparison results to Excel in 'Validation' folder in the parent directory"""
    # Create Validation directory in parent if it doesn't exist
    validation_dir = os.path.join(os.path.dirname(os.getcwd()), "Validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    if filename is None:
        # Use first dataset name in results
        dataset_name = results_df['Dataset'].iloc[0]
        filename = f'comparison_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    
    # Full path in Validation directory
    full_path = os.path.join(validation_dir, filename)
    
    results_df.to_excel(full_path, index=False)
    print(f"Results saved to {full_path}")
    return full_path

# %% Run comparison

# Extract dataset name from file path
dataset_name = os.path.basename(file_path).replace('.xlsx', '')

# Run comparison with dataset name
comparison_results, ml_result, lbfgs_result, huck_result = compare_methods(
    fatigue_data, 
    ref_values, 
    dataset_name=dataset_name
)

# %%

save_comparison_to_excel(comparison_results)

# %% [markdown] about fmin() output 
# ## SciPy fmin optimization output
# 
# PyLife's Woehler module uses SciPy's `optimize.fmin()` function (Nelder-Mead algorithm) in MaxLikeInf to determine SD (endurance limit) and TS (scatter).
# 
# [Scipy fmin output](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html)
#
# ### PyLife's original MaxLikeInf implementation (__max_likelihood_inf_limit method)
# ```python
# SD_start = self._fd.fatigue_limit
# TS_start = 1.2

# var_opt = optimize.fmin(
#     lambda p: -self._lh.likelihood_infinite(p[0], p[1]),
#     [SD_start, TS_start], 
#     disp=False,  # Note: PyLife suppresses output
#     full_output=True
# )
# extracts values withoutchecking warnflags
# SD_50 = var_opt[0][0]
# TS = var_opt[0][1]
#
# return SD_50, TS
# 
# # likelihood_infinite(self, SD, TS): 
# Calculates likelihood for points in the infinite zone (horizontal part of curve)
# 
# infinite_zone = self._fd.infinite_zone
# std_log = scattering_range_to_std(TS)
# t = np.logical_not(self._fd.infinite_zone.fracture).astype(np.float64)
# likelihood = stats.norm.cdf(np.log10(infinite_zone.load/SD), scale=abs(std_log))
# non_log_likelihood = t+(1.-2.*t)*likelihood
# if non_log_likelihood.eq(0.0).any():
#     return -np.inf
# return np.log(non_log_likelihood).sum()
# ```
#
# When using `full_output=True`, fmin returns a tuple where:
# - `var_opt[0]` = optimized parameters (SD and TS)
# - `var_opt[1]` = final optimized function value
# - `var_opt[2]` = number of function evaluations
# - `var_opt[3]` = number of iterations
# - `var_opt[4]` = warnflag (0=success, 1=max iterations, 2=function not improving)
# - `var_opt[5]` = termination message
# 
# The warnflag at index 4 is the critical value for determining success:
# - warnflag = 0: Optimization succeeded (converged properly)
# - warnflag = 1: Maximum number of iterations reached without convergence
# - warnflag = 2: Function evaluations not changing (possible precision loss)


# %% Test PyLife's infinite zone definition

# Load a test file and prepare data
file_path = "All Data/4PB_2.xlsx"  # Change to a file that gives good results
df_test, ref_values = analyze_fatigue_file(file_path)
df_prepared = df_test[['load', 'cycles', 'censor']]
df_prepared = woehler.determine_fractures(df_prepared, NG)
fatigue_data = df_prepared.fatigue_data

# Run Elementary analysis to get ND
elementary_result = woehler.Elementary(fatigue_data).analyze()
ND_value = elementary_result.ND

print(f"Dataset: {os.path.basename(file_path)}")
print(f"Transition (ND) cycles: {ND_value:.0f}")

# Run MaxLikeInf analysis to compare
maxlike_result = woehler.MaxLikeInf(fatigue_data).analyze()
print(f"MaxLikeInf SD: {maxlike_result.SD:.2f}")
print(f"MaxLikeInf ND: {maxlike_result.ND:.0f}")

# Get points with cycles >= ND (staircase region)
staircase_region = df_prepared[df_prepared['cycles'] >= ND_value]
print(f"\nPoints in staircase region (cycles >= ND): {len(staircase_region)}")
print("Load levels in staircase region:", sorted(staircase_region['load'].unique()))
print("Number of failures:", sum(staircase_region['fracture']))
print("Number of runouts:", sum(~staircase_region['fracture']))

# Get points with cycles < ND (slope region)
slope_region = df_prepared[df_prepared['cycles'] < ND_value]
print(f"\nPoints in slope region (cycles < ND): {len(slope_region)}")
print("Load levels in slope region:", sorted(slope_region['load'].unique()))
print("Number of failures:", sum(slope_region['fracture']))
print("Number of runouts:", sum(~slope_region['fracture']))

# Calculate load range in staircase region
if not staircase_region.empty:
    staircase_min_load = staircase_region['load'].min()
    staircase_max_load = staircase_region['load'].max()
    print(f"\nStaircase region load range: {staircase_min_load:.2f} to {staircase_max_load:.2f}")
    
    # Count failures at each load level in staircase region
    staircase_failures = staircase_region.groupby('load')['fracture'].sum()
    print("\nFailures at each load level in staircase region:")
    for load, failures in staircase_failures.items():
        total = len(staircase_region[staircase_region['load'] == load])
        runouts = total - failures
        print(f"Load {load:.2f}: {failures} failures, {runouts} runouts")


# %% Batch process all files in a directory with automatic runout detection
def batch_compare_methods_in_directory(directory_path, default_ng=5000000):
    """Process all Excel files in a directory, comparing MaxLikeInf and Huck methods
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing Excel files
    default_ng : int, optional
        Default NG value to use if automatic detection fails
    """
    # Find all Excel files in the directory
    excel_files = [f for f in os.listdir(directory_path) 
                  if f.endswith('.xlsx') and not f.startswith('~$')]
    
    print(f"Found {len(excel_files)} Excel files in {directory_path}")
    
    all_results = []
    
    for file_name in excel_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            print(f"\n\n=== Processing {file_name} ===")
            
            # Load data
            df_test, ref_values = analyze_fatigue_file(file_path)
            
            # Automatically determine NG from survivors
            survivors = df_test[df_test['censor'] == 0]
            if not survivors.empty:
                # Get the minimum cycles for survivors and round to nearest 10000
                lowest_survivor = survivors['cycles'].min()
                ng = round(lowest_survivor / 10000) * 10000
                print(f"Automatically detected NG: {ng:,} cycles")
            else:
                ng = default_ng
                print(f"No survivors detected, using default NG: {ng:,} cycles")
            
            # Prepare data
            df_prepared = df_test[['load', 'cycles', 'censor']]
            df_prepared = woehler.determine_fractures(df_prepared, ng)
            fatigue_data = df_prepared.fatigue_data
            
            # Run comparison
            comparison_df, ml_result, lbfgs_result, huck_result = compare_methods(fatigue_data, ref_values)
            
            # Calculate derived parameters
            ml_slog = np.log10(ml_result.TS)/2.5361
            huck_slog = np.log10(huck_result.TS)/2.5361
            
            # Create result entry
            file_results = {
                'file': file_name,
                'ng_value': ng,
                'nm_SD': ml_result.SD,
                'nm_TS': ml_result.TS,
                'nm_slog': ml_slog,
                'huck_SD': huck_result.SD,
                'huck_TS': huck_result.TS,
                'huck_slog': huck_slog,
                'lbfgs_used': lbfgs_result is not None
            }
            
            # Add L-BFGS-B results if available
            if lbfgs_result is not None:
                lbfgs_slog = np.log10(lbfgs_result.TS)/2.5361
                file_results.update({
                    'lbfgs_SD': lbfgs_result.SD,
                    'lbfgs_TS': lbfgs_result.TS,
                    'lbfgs_slog': lbfgs_slog
                })
            
            # Add reference values and diffs if available
            if ref_values is not None and 'Pü50' in ref_values:
                file_results.update({
                    'ref_SD': ref_values['Pü50'],
                    'ref_slog': ref_values.get('slog', None),
                    'nm_SD_diff_pct': (ml_result.SD - ref_values['Pü50'])/ref_values['Pü50'] * 100,
                    'huck_SD_diff_pct': (huck_result.SD - ref_values['Pü50'])/ref_values['Pü50'] * 100
                })
                
                if lbfgs_result is not None:
                    file_results['lbfgs_SD_diff_pct'] = (lbfgs_result.SD - ref_values['Pü50'])/ref_values['Pü50'] * 100
            
            all_results.append(file_results)
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            traceback.print_exc()
    
    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'method_comparison_{os.path.basename(directory_path)}_{timestamp}.xlsx'
        results_df.to_excel(output_filename, index=False)
        print(f"\nResults saved to {output_filename}")
        
        # Display results
        from IPython.display import display
        display(results_df)
        
        return results_df
    else:
        print("No results were successfully processed!")
        return None

# Example usage:
# batch_results = batch_compare_methods_in_directory("All Data")