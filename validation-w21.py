# %% Settings and Imports

import os
import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import optimize, stats
from datetime import datetime
import traceback

# PyLife imports
import pylife.materialdata.woehler as woehler
from pylife.materialdata.woehler.likelihood import Likelihood

# Import custom classes
from huck import HuckMethod


# %% GLOBAL CONFIGURATION - Update these for analysis

# Dataset settings
DATASET_PATH = "All Data/4PB_2.xlsx" # Path to Excel file
NG = 5000000 # Runout cycles (cycles to infinite life)
N_LCF = 10000 # LCF pivot point

# Optimization bounds
MANUAL_TS_BOUNDS = None # None = use Huck's TS, or tuple like (1.0, 10.0)
SD_BOUNDS = None # Will be calculated as (min_load * 0.5, max_load * 2.0)

# Output settings
OUTPUT_BASE_DIR = "Validation_Output"
COMPARISON_FILENAME = "method_comparison_results.xlsx"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

print("Settings loaded successfully!")
print(f"Dataset: {DATASET_PATH}")
print(f"NG: {NG:,} cycles")
print(f"Manual TS bounds: {MANUAL_TS_BOUNDS}")
print(f"Output directory: {OUTPUT_BASE_DIR}")


# %% Data Loading and Preparation
def load_and_prepare_data(file_path, ng):
    """Load fatigue data and prepare for analysis"""
    try:
        # Load data
        df_test = pd.read_excel(file_path, sheet_name='Data')
        
        # Rename 'loads' to 'load' if needed
        if 'loads' in df_test.columns:
            df_test = df_test.rename(columns={'loads': 'load'})
        
        # Prepare data for PyLife
        df_prepared = df_test[['load', 'cycles', 'censor']].copy()
        df_prepared = woehler.determine_fractures(df_prepared, ng)
        fatigue_data = df_prepared.fatigue_data
        
        # Calculate dynamic SD bounds
        min_load = fatigue_data.load.min()
        max_load = fatigue_data.load.max()
        sd_bounds = (min_load * 0.5, max_load * 2.0)
        avg_sd = (min_load + max_load) / 2
        
        fatigue_data.avg_sd = avg_sd
        
        print(f"Data loaded successfully!")
        print(f"Total data points: {len(df_prepared)}")
        print(f"Load range: {min_load:.1f} to {max_load:.1f}")
        print(f"Calculated SD bounds: {sd_bounds}")
        print(f"Average SD: {avg_sd:.1f}")
        
        
        # Initialize global results dictionary only if new dataset
        global ANALYSIS_RESULTS
        current_dataset_path = file_path
        existing_path = ANALYSIS_RESULTS.get('dataset_info', {}).get('path', None)

        if existing_path != current_dataset_path:
            # New dataset - reset results
            ANALYSIS_RESULTS = {
                'dataset_info': {
                    'path': file_path,
                    'ng': ng,
                    'total_points': fatigue_data.num_tests,
                    'load_range': (min_load, max_load)
                }
            }
            print(f"New dataset - global results dictionary initialized")
        else:
            print(f"Same dataset - keeping existing results")
        
        try:
            df_ref = pd.read_excel(file_path, sheet_name='Jurojin_results', header=None)
            ref_values = {}
            for idx, row in df_ref.iterrows():
                param_name = row[0]
                param_value = row[1]
                if isinstance(param_value, str) and ',' in param_value:
                    param_value = float(param_value.replace(',', '.'))
                ref_values[param_name] = param_value
            print(f"Jurojin reference values extracted")
        except:
            ref_values = None
            print(f"No Jurojin reference values found")

        # Run Huck's method automatically
        try:
            huck_analyzer = HuckMethod(fatigue_data)
            huck_result = huck_analyzer.analyze()
            huck_slog = np.log10(huck_result.TS) / 2.5361
            print(f"Huck's method completed: SD={huck_result.SD:.2f}")
        except:
            huck_result = None
            print(f"Huck's method failed")

        # Store Jurojin reference in global results
        if ref_values is not None:
            ANALYSIS_RESULTS['Jurojin_Reference'] = {
                'Method': 'Jurojin_Reference',
                'SD': ref_values.get('Pü50', None),
                'TS': ref_values.get('TS', None), 
                'slog': ref_values.get('slog', None),
                'ND': ref_values.get('ND', None),
                'k_1': ref_values.get('k', None),
                'TN': None,
                'status_message': 'Reference values',
                'iterations': 'N/A',
                'ts_source': 'Reference'
            }

        # Store Huck's results in global results
        if huck_result is not None:
            ANALYSIS_RESULTS['Huck'] = {
                'Method': 'Huck',
                'SD': huck_result.SD,
                'TS': huck_result.TS,
                'slog': huck_slog,
                'ND': huck_result.ND,
                'k_1': huck_result.k_1,
                'TN': huck_result.TN,
                'status_message': 'Staircase analysis',
                'iterations': 'N/A',
                'ts_source': 'Huck'
            }
        
        
        return fatigue_data, sd_bounds, df_prepared
        
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None, None

    
# %% Method 1: MaxLikeInf Analysis with Tracking
class MaxLikeInfWithTracking(woehler.MaxLikeInf):
    """MaxLikeInf with optimization step tracking"""
    
    def __init__(self, fatigue_data):
        super().__init__(fatigue_data)
        self.optimization_steps = []
        
    def _MaxLikeInf__max_likelihood_inf_limit(self):
        """Override to add tracking"""
        SD_start = self._fd.fatigue_limit
        TS_start = 1.2
        
        def tracked_objective(p):
            likelihood = self._lh.likelihood_infinite(p[0], p[1])
            self.optimization_steps.append({
                'Step': len(self.optimization_steps) + 1,
                'SD': p[0],
                'TS': p[1],
                'Likelihood': likelihood
            })
            return -likelihood
        
        result = optimize.fmin(tracked_objective, [SD_start, TS_start], 
                              disp=False, full_output=True)
        
        # Store convergence info
        self.final_values = result[0]
        self.warnflag = result[4]
        self.message = result[5] if len(result) > 5 else "No message"
        
        return result[0][0], result[0][1]

def run_maxlike_analysis(fatigue_data):
    """Run MaxLikeInf analysis with tracking and return results"""
    
    # Run analysis with tracking
    analyzer = MaxLikeInfWithTracking(fatigue_data)
    result = analyzer.analyze()
    
    # Calculate slog
    slog = np.log10(result.TS) / 2.5361
    
    # Get convergence info
    warnflag_meanings = {
        0: "Success - optimization converged",
        1: "Maximum number of iterations reached",
        2: "Function values not changing (precision loss)",
        3: "NaN result encountered"
    }
    status_message = warnflag_meanings.get(analyzer.warnflag, "Unknown")
    
    # Prepare results dictionary for saving
    results_dict = {
        'Method': 'MaxLikeInf',
        'SD': result.SD,
        'TS': result.TS,
        'slog': slog,
        'ND': result.ND,
        'k_1': result.k_1,
        'TN': result.TN,
        'warnflag': analyzer.warnflag,
        'status_message': status_message,
        'iterations': len(analyzer.optimization_steps),
        'optimization_steps': analyzer.optimization_steps,
        'result_object': result  # Store for plotting
    }
    
    return results_dict


# %% Method 2: L-BFGS-B Analysis
def run_lbfgsb_analysis(fatigue_data, ts_bounds=None):
    """Run L-BFGS-B analysis with optional TS bounds or Huck's TS as default"""
    
    # Determine TS handling approach
    if ts_bounds is None:
        # Use Huck's method for deterministic TS
        print("No TS bounds provided. Using Huck's method for deterministic TS...")
        huck_analyzer = HuckMethod(fatigue_data)
        huck_result = huck_analyzer.analyze()
        fixed_ts = huck_result.TS
        print(f"Huck's TS: {fixed_ts:.4f}")
        
        # Optimization setup: only optimize SD
        optimization_steps = []
        lh = Likelihood(fatigue_data)
        
        def objective_function_fixed_ts(sd_array):
            sd = sd_array[0]
            likelihood = lh.likelihood_infinite(sd, fixed_ts)
            optimization_steps.append({
                'Step': len(optimization_steps) + 1,
                'SD': sd,
                'TS': fixed_ts,
                'Likelihood': likelihood
            })
            return -likelihood
        
        # Run optimization with only SD bounds
        sd_bounds = [(SD_BOUNDS[0], SD_BOUNDS[1])]
        initial_sd = fatigue_data.fatigue_limit
        
        result = optimize.minimize(
            objective_function_fixed_ts,
            [initial_sd],
            method='L-BFGS-B',
            bounds=sd_bounds
        )
        
        final_sd = result.x[0]
        final_ts = fixed_ts
        ts_source = "Huck"
        
    else:
        # Optimize both SD and TS within provided bounds
        print(f"Using manual TS bounds: {ts_bounds}")
        
        # Optimization setup: optimize both SD and TS
        optimization_steps = []
        lh = Likelihood(fatigue_data)
        
        def objective_function_both(params):
            sd, ts = params
            likelihood = lh.likelihood_infinite(sd, ts)
            optimization_steps.append({
                'Step': len(optimization_steps) + 1,
                'SD': sd,
                'TS': ts,
                'Likelihood': likelihood
            })
            return -likelihood
        
        # Set up bounds for both parameters
        bounds = [SD_BOUNDS, ts_bounds]
        initial_params = [fatigue_data.fatigue_limit, 1.2]
        
        result = optimize.minimize(
            objective_function_both,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        final_sd, final_ts = result.x
        ts_source = "Manual"
    
    # Calculate additional parameters
    slog = np.log10(final_ts) / 2.5361
    
    # Recalculate ND using optimized SD (consistent with MaxLikeInf approach)
    # Need to get Elementary result for slope calculation
    elementary_analyzer = woehler.Elementary(fatigue_data)
    elementary_result = elementary_analyzer.analyze()
    
    # Calculate ND using the transition_cycles method approach
    slope = elementary_result.k_1
    # Using the same formula as in elementary.py _transition_cycles
    lg_intercept = np.log10(elementary_result.ND) - (-slope) * np.log10(elementary_result.SD)
    final_nd = 10**(lg_intercept + (-slope) * np.log10(final_sd))
    
    # Get optimization status message
    status_msg = "Success" if result.success else f"Failed: {result.message}"
    
    # Prepare results dictionary
    results_dict = {
        'Method': 'L-BFGS-B',  # Consistent method name
        'SD': final_sd,
        'TS': final_ts,
        'slog': slog,
        'ND': final_nd,
        'k_1': slope,
        'TN': elementary_result.TN,
        'optimization_success': result.success,
        'status_message': status_msg,
        'iterations': len(optimization_steps),
        'function_evaluations': result.nfev if hasattr(result, 'nfev') else 'N/A',
        'optimization_steps': optimization_steps,
        'ts_source': ts_source,  # Track whether TS from Huck or Manual
        'result_object': pd.Series({
            'SD': final_sd, 'TS': final_ts, 'ND': final_nd, 
            'k_1': slope, 'TN': elementary_result.TN
        })
    }
    
    return results_dict


# %% Method 3: MaxLikeFull Analysis  
def run_maxlikefull_analysis(fatigue_data, ts_bounds=None):
    """Run MaxLikeFull analysis with optional TS bounds or Huck's TS as default"""
    
    # Determine TS handling approach
    if ts_bounds is None:
        # Use Huck's method for deterministic TS
        print("No TS bounds provided. Using Huck's method for deterministic TS...")
        huck_analyzer = HuckMethod(fatigue_data)
        huck_result = huck_analyzer.analyze()
        fixed_ts = huck_result.TS
        print(f"Huck's TS: {fixed_ts:.4f}")
        
        # Run MaxLikeFull with fixed TS
        analyzer = woehler.MaxLikeFull(fatigue_data)
        result = analyzer.analyze(fixed_parameters={'TS': fixed_ts})
        ts_source = "Huck"
        
    else:
        # Let MaxLikeFull optimize both SD and TS within bounds
        print(f"Using manual TS bounds: {ts_bounds}")
        print("Note: MaxLikeFull doesn't directly support TS bounds - optimizing freely")
        
        # Run standard MaxLikeFull (optimizes all parameters)
        analyzer = woehler.MaxLikeFull(fatigue_data)
        result = analyzer.analyze()
        ts_source = "Manual"
    
    # Calculate slog
    slog = np.log10(result.TS) / 2.5361
    
    # Get status message
    status_msg = "Success"  # MaxLikeFull doesn't return detailed status
    
    # Prepare results dictionary
    results_dict = {
        'Method': 'MaxLikeFull',  # Consistent method name
        'SD': result.SD,
        'TS': result.TS,
        'slog': slog,
        'ND': result.ND,
        'k_1': result.k_1,
        'TN': result.TN,
        'status_message': status_msg,
        'iterations': 'N/A',  # No tracking available
        'optimization_steps': [],  # Empty for compatibility
        'ts_source': ts_source,  # Track whether TS from Huck or Manual
        'result_object': result
    }
    
    return results_dict


# %% Method 4: Nelder-Mead Analysis with minimize + likelihood override

# %% Method 4: Nelder-Mead with std_log

# Custom Likelihood class to work with std_log directly
class StdLogLikelihood(Likelihood):
    """Custom Likelihood class that works with std_log instead of TS"""
    
    def __init__(self, fatigue_data):
        super().__init__(fatigue_data)
    
    def likelihood_infinite(self, SD, std_log):
        """Override to work with std_log directly instead of TS"""
        infinite_zone = self._fd.infinite_zone
        t = np.logical_not(self._fd.infinite_zone.fracture).astype(np.float64)
        likelihood = stats.norm.cdf(np.log10(infinite_zone.load/SD), scale=abs(std_log))
        non_log_likelihood = t+(1.-2.*t)*likelihood
        if non_log_likelihood.eq(0.0).any():
            return -np.inf

        return np.log(non_log_likelihood).sum()

def run_nelder_mead_stdlog_analysis(fatigue_data):
    """Run Nelder-Mead analysis using std_log instead of TS"""
    
    # Create custom likelihood object
    custom_lh = StdLogLikelihood(fatigue_data)
    
    # Initial values
    initial_sd = fatigue_data.avg_sd  # Use avg_sd as starting point
    initial_std_log = 1.0            # Start with slog=1 as requested
    
    print(f"Using Nelder-Mead with std_log - Initial SD: {initial_sd:.2f}, Initial std_log: {initial_std_log:.2f}")
    
    # Setup optimization tracking
    optimization_steps = []
    
    # Define objective function with tracking
    def tracked_objective(params):
        sd, std_log = params
        likelihood = custom_lh.likelihood_infinite(sd, std_log)
        
        # Convert std_log to TS for tracking and later comparison
        ts_value = 10**(2.5361 * std_log)
        
        optimization_steps.append({
            'Step': len(optimization_steps) + 1,
            'SD': sd,
            'std_log': std_log,
            'TS': ts_value,  # Calculate equivalent TS
            'Likelihood': likelihood
        })
        
        return -likelihood
    
    # Run Nelder-Mead optimization using minimize()
    initial_params = [initial_sd, initial_std_log]
    result = optimize.minimize(
        tracked_objective,
        initial_params,
        method='Nelder-Mead',
        options={'disp': False}  # Set to True for detailed output
    )
    
    # Extract results
    final_sd = result.x[0]
    final_std_log = result.x[1]
    final_ts = 10**(2.5361 * final_std_log)  # Convert to TS for compatibility
    
    # Calculate ND using Elementary parameters
    elementary_analyzer = woehler.Elementary(fatigue_data)
    elementary_result = elementary_analyzer.analyze()
    
    # Calculate ND using transition_cycles approach
    slope = elementary_result.k_1
    lg_intercept = np.log10(elementary_result.ND) - (-slope) * np.log10(elementary_result.SD)
    final_nd = 10**(lg_intercept + (-slope) * np.log10(final_sd))
    
    # Create a Series for compatibility with other methods
    result_series = pd.Series({
        'SD': final_sd,
        'TS': final_ts,
        'ND': final_nd,
        'k_1': slope,
        'TN': elementary_result.TN,
        'slog': final_std_log  # Store the optimized std_log
    })
    
    # Get optimization metadata
    status_message = f"Success: {result.success}, Message: {result.message}"
    
    # Prepare results dictionary
    results_dict = {
        'Method': 'Nelder-Mead-StdLog',
        'SD': final_sd,
        'TS': final_ts,
        'slog': final_std_log,
        'ND': final_nd,
        'k_1': slope,
        'TN': elementary_result.TN,
        'optimization_success': result.success,
        'status_message': status_message,
        'iterations': result.nit if hasattr(result, 'nit') else None,
        'function_evaluations': result.nfev,
        'optimization_steps': optimization_steps,
        'result_object': result_series  # Store for plotting
    }
    
    # Print summary
    print("\nNelder-Mead-StdLog Results:")
    print(f"SD: {final_sd:.2f}")
    print(f"std_log: {final_std_log:.4f} (equivalent TS: {final_ts:.2f})")
    print(f"ND: {final_nd:.0f}")
    print(f"Status: {status_message}")
    print(f"Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
    
    return results_dict


# %% Plotting functions
def display_sn_curve(fatigue_data, result, method_name):
    """Display SN curve in Jupyter"""
    df = fatigue_data._obj
    failures = df[df['fracture']]
    survivors = df[~df['fracture']]
    
    fig = go.Figure()
    
    # Plot data points
    if not failures.empty:
        fig.add_trace(go.Scatter(
            x=failures['cycles'], y=failures['load'],
            mode='markers', marker=dict(color='red', symbol='cross', size=8),
            name='Failures'
        ))
    
    if not survivors.empty:
        fig.add_trace(go.Scatter(
            x=survivors['cycles'], y=survivors['load'],
            mode='markers', marker=dict(color='blue', symbol='triangle-right', size=8),
            name='Survivors'
        ))
    
    # Plot SN curve
    if not np.isnan(result.ND):
        min_cycles = df['cycles'].min()
        L_LCF = 10**(np.log10(result.SD) - (np.log10(result.ND/min_cycles))/-result.k_1)
        
        fig.add_trace(go.Scatter(
            x=[min_cycles, result.ND],
            y=[L_LCF, result.SD],
            mode='lines', line=dict(color='green', width=2),
            name=f'{method_name} (LCF)'
        ))
        
        fig.add_trace(go.Scatter(
            x=[result.ND, NG],
            y=[result.SD, result.SD],
            mode='lines', line=dict(color='green', width=2, dash='dash'),
            name=f'{method_name} (HCF)'
        ))
    
    fig.update_layout(
        title=f'SN Curve - {method_name}',
        xaxis_title='Cycles', yaxis_title='Load',
        xaxis_type="log", yaxis_type="log",
        width=800, height=600
    )
    
    fig.show()

def display_convergence_plot(optimization_steps, method_name):
    """Display convergence plot in Jupyter"""
    df_steps = pd.DataFrame(optimization_steps)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['SD Convergence', 'TS Convergence'])
    
    # SD convergence  
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['SD'],
                            mode='lines+markers', name='SD'),
                  row=1, col=1)
    
    # TS convergence
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['TS'],
                            mode='lines+markers', name='TS'),
                  row=1, col=2)
    
    fig.update_layout(title=f'{method_name} Convergence', height=400, showlegend=False)
    fig.show()

    
# %% Save Results to File
def save_method_results(results_dict, fatigue_data, output_base_dir):
    """Save analysis results, plots to timestamped directory"""
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = results_dict['Method']
    method_dir = os.path.join(output_base_dir, f"{method_name}_{timestamp}")
    os.makedirs(method_dir, exist_ok=True)
    
    # Prepare results for CSV (exclude non-serializable items)
    csv_results = {k: v for k, v in results_dict.items() 
                  if k not in ['optimization_steps', 'result_object']}
    
    # Save results to CSV
    results_df = pd.DataFrame([csv_results])
    results_df.to_csv(os.path.join(method_dir, 'results.csv'), index=False)
    
    # Save SN curve
    save_sn_curve(fatigue_data, results_dict['result_object'], method_dir, method_name)
    
    # Save convergence plot (if optimization steps exist)
    if 'optimization_steps' in results_dict:
        save_convergence_plot(results_dict['optimization_steps'], method_dir, method_name)
    
    print(f"Results saved to: {method_dir}")
    return method_dir

def save_sn_curve(fatigue_data, result, output_dir, method_name):
    """Save SN curve plot to file"""
    df = fatigue_data._obj
    failures = df[df['fracture']]
    survivors = df[~df['fracture']]
    
    fig = go.Figure()
    
    if not failures.empty:
        fig.add_trace(go.Scatter(
            x=failures['cycles'], y=failures['load'],
            mode='markers', marker=dict(color='red', symbol='cross', size=8),
            name='Failures'
        ))
    
    if not survivors.empty:
        fig.add_trace(go.Scatter(
            x=survivors['cycles'], y=survivors['load'],
            mode='markers', marker=dict(color='blue', symbol='triangle-right', size=8),
            name='Survivors'
        ))
    
    if not np.isnan(result.ND):
        min_cycles = df['cycles'].min()
        L_LCF = 10**(np.log10(result.SD) - (np.log10(result.ND/min_cycles))/-result.k_1)
        
        fig.add_trace(go.Scatter(
            x=[min_cycles, result.ND],
            y=[L_LCF, result.SD],
            mode='lines', line=dict(color='green', width=2),
            name=f'{method_name} (LCF)'
        ))
        
        fig.add_trace(go.Scatter(
            x=[result.ND, NG],
            y=[result.SD, result.SD],
            mode='lines', line=dict(color='green', width=2, dash='dash'),
            name=f'{method_name} (HCF)'
        ))
    
    fig.update_layout(
        title=f'SN Curve - {method_name}',
        xaxis_title='Cycles', yaxis_title='Load',
        xaxis_type="log", yaxis_type="log",
        width=800, height=600
    )
    
    fig.write_image(os.path.join(output_dir, 'sn_curve.png'))

def save_convergence_plot(optimization_steps, output_dir, method_name):
    """Save convergence plot to file"""
    df_steps = pd.DataFrame(optimization_steps)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['SD Convergence', 'TS Convergence'])
    
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['SD'],
                            mode='lines+markers', name='SD'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['TS'],
                            mode='lines+markers', name='TS'),
                  row=1, col=2)
    
    fig.update_layout(title=f'{method_name} Convergence', height=400, showlegend=False)
    fig.write_image(os.path.join(output_dir, 'convergence.png'))


# %% Compile Comparison Results to Excel
def compile_results_to_excel(analysis_results, filename=None, output_base_dir=OUTPUT_BASE_DIR):
    """Compile all method results into comparison Excel file"""
    
    # Create Results_Comparison directory
    comparison_dir = os.path.join(output_base_dir, "Results_Comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Default filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.xlsx"
    
    # Ensure .xlsx extension
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    filepath = os.path.join(comparison_dir, filename)
    
    # Extract dataset info
    dataset_info = analysis_results.get('dataset_info', {})
    dataset_name = os.path.basename(dataset_info.get('path', 'Unknown'))
    
    # Compile results from all methods
    comparison_rows = []
    for method_name, method_results in analysis_results.items():
        if method_name == 'dataset_info':  # Skip dataset info
            continue
            
        row = {
            'Dataset': dataset_name,
            'Method': method_name,
            'SD': method_results.get('SD'),
            'TS': method_results.get('TS'),
            'slog': method_results.get('slog'),
            'ND': method_results.get('ND'),
            'k_1': method_results.get('k_1'),
            'TN': method_results.get('TN'),
            'Status': method_results.get('status_message', 'N/A'),
            'Iterations': method_results.get('iterations', 'N/A'),
            'NG': dataset_info.get('ng'),
            'Total_Points': dataset_info.get('total_points'),
            'Timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        comparison_rows.append(row)
    
    if not comparison_rows:
        print("No method results to compile!")
        return None
    
    # Create DataFrame
    new_df = pd.DataFrame(comparison_rows)
    
    # Check if file exists
    if os.path.exists(filepath):
        # Append to existing file
        try:
            existing_df = pd.read_excel(filepath)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"Appending {len(new_df)} rows to existing file")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Creating new file instead")
            combined_df = new_df
    else:
        # Create new file
        combined_df = new_df
        print(f"Creating new comparison file")
    
    # Save to Excel
    try:
        combined_df.to_excel(filepath, index=False)
        print(f"Results compiled to: {filepath}")
        print(f"Total rows in file: {len(combined_df)}")
        return filepath
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        return None


# %% File Selection, Data Loading
# Update these settings and run this block
DATASET_PATH = "All Data/4PB_7.xlsx"  # Update this path
NG = 5000000                          # Update if needed

# Load and prepare data
fatigue_data, calculated_sd_bounds, df_prepared = load_and_prepare_data(DATASET_PATH, NG)

# %% Select Method, run Analysis

if fatigue_data is not None:
    # Update global SD_BOUNDS for other methods
    SD_BOUNDS = calculated_sd_bounds
    
    # Method selection: Uncomment to select Method to run
    
    # METHOD_TO_RUN = "MaxLikeInf"
    # METHOD_TO_RUN = "L-BFGS-B"
    # METHOD_TO_RUN = "MaxLikeFull"
    METHOD_TO_RUN = "Nelder-Mead-StdLog"

    # Run selected analysis
    if METHOD_TO_RUN == "MaxLikeInf":
        method_results = run_maxlike_analysis(fatigue_data)
        
    elif METHOD_TO_RUN == "L-BFGS-B":
        # Default: uses Huck's TS. Manual TS bounds e.g.: ts_bounds=(1.0, 10.0)
        method_results = run_lbfgsb_analysis(fatigue_data)
        
    elif METHOD_TO_RUN == "MaxLikeFull":
        # Default: uses Huck's TS. Manual TS bounds e.g.: ts_bounds=(1.0, 10.0)
        method_results = run_maxlikefull_analysis(fatigue_data)
        
    elif METHOD_TO_RUN == "Nelder-Mead-StdLog":
        method_results = run_nelder_mead_stdlog_analysis(fatigue_data)
        
    else:
        print(f"Unknown method: {METHOD_TO_RUN}")
        method_results = None
    
    if method_results is not None:
        # Store in global results dictionary
        ANALYSIS_RESULTS[method_results['Method']] = method_results
        
        # Display results
        print(f"\n=== {method_results['Method']} Results ===")
        print(f"SD: {method_results['SD']:.2f}")
        print(f"TS: {method_results['TS']:.3f}")
        print(f"slog: {method_results['slog']:.4f}")
        print(f"ND: {method_results['ND']:.0f}")
        print(f"k_1: {method_results['k_1']:.3f}")
        print(f"Status: {method_results['status_message']}")
        print(f"Iterations: {method_results['iterations']}")
        
        # Display SN curve and convergence plot
        display_sn_curve(fatigue_data, method_results['result_object'], method_results['Method'])
        display_convergence_plot(method_results['optimization_steps'], method_results['Method'])
        
        print(f"\nAnalysis complete. Results ready for saving.")
    else:
        print("Analysis failed.")
else:
    print("Failed to load data. Check file path and format.")
    

# %% Save last run Method

# Save selected method results
if 'ANALYSIS_RESULTS' in globals() and METHOD_TO_RUN in ANALYSIS_RESULTS:
    save_method_results(ANALYSIS_RESULTS[METHOD_TO_RUN], fatigue_data, OUTPUT_BASE_DIR)
    print(f"Saved {METHOD_TO_RUN} results")
else:
    print(f"No {METHOD_TO_RUN} results to save. Run analysis first.")


# %% Compile current analysis results to Excel
# Update the filename as needed, or leave None for automatic timestamp
COMPARISON_FILENAME = None  # or "my_campaign_results.xlsx"

if 'ANALYSIS_RESULTS' in globals() and len(ANALYSIS_RESULTS) > 1:  # More than just dataset_info
    compile_results_to_excel(ANALYSIS_RESULTS, COMPARISON_FILENAME)
else:
    print("No results to compile. Run analysis first.")


# %%
print("Current ANALYSIS_RESULTS:")
for method_name, results in ANALYSIS_RESULTS.items():
    if method_name != 'dataset_info':
        print(f"  {method_name}: SD={results['SD']:.2f}")
        
    
# %% Run All Methods - Complete Analysis and Save
# Update dataset path and run this block to analyze with all methods
DATASET_PATH = "All Data/NO35_gekerbt.xlsx"  # Update this path
NG = 4000000                          # Update if needed

# Load and prepare data
print("=== Loading Dataset ===")
fatigue_data, calculated_sd_bounds, df_prepared = load_and_prepare_data(DATASET_PATH, NG)

if fatigue_data is not None:
    # Update global SD_BOUNDS for optimization methods
    SD_BOUNDS = calculated_sd_bounds
    
    # List of methods to run
    methods_to_run = ["MaxLikeInf", "L-BFGS-B", "MaxLikeFull"]
    
    print(f"\n=== Running All Methods ===")
    for method_name in methods_to_run:
        print(f"\n--- Running {method_name} ---")
        
        # Run selected analysis
        if method_name == "MaxLikeInf":
            method_results = run_maxlike_analysis(fatigue_data)
            
        elif method_name == "L-BFGS-B":
            # Uses Huck's TS by default
            method_results = run_lbfgsb_analysis(fatigue_data)
            
        elif method_name == "MaxLikeFull":
            # Uses Huck's TS by default  
            method_results = run_maxlikefull_analysis(fatigue_data)
        
        # Store in global results dictionary
        ANALYSIS_RESULTS[method_results['Method']] = method_results
        
        # Display brief results
        print(f"  SD: {method_results['SD']:.2f}")
        print(f"  TS: {method_results['TS']:.3f}")
        print(f"  Status: {method_results['status_message']}")
        
        # Save method results to directory
        save_method_results(method_results, fatigue_data, OUTPUT_BASE_DIR)
        print(f"  Results saved to directory")
    
    # Compile all results to Excel
    print(f"\n=== Compiling Results to Excel ===")
    compile_results_to_excel(ANALYSIS_RESULTS, filename="all_methods_comparison.xlsx")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Methods run: {list(ANALYSIS_RESULTS.keys())}")
    print(f"All results saved and compiled!")
    
else:
    print("Failed to load data. Check file path and format.")

# %%
