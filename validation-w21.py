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
        
        print(f"Data loaded successfully!")
        print(f"Total data points: {len(df_prepared)}")
        print(f"Load range: {min_load:.1f} to {max_load:.1f}")
        print(f"Calculated SD bounds: ({sd_bounds[0]:.1f}, {sd_bounds[1]:.1f})")
        
        return fatigue_data, sd_bounds, df_prepared
        
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None, None

# %% Load the data
fatigue_data, calculated_sd_bounds, df_prepared = load_and_prepare_data(DATASET_PATH, NG)

if fatigue_data is not None:
    # Update global SD_BOUNDS
    SD_BOUNDS = calculated_sd_bounds
    print(f"Ready for analysis with {fatigue_data.num_tests} test points")
else:
    print("Failed to load data. Check file path and format.")
    
    
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
    """Run MaxLikeInf analysis with tracking and display results"""
    
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
    
    # Display results
    print(f"\n=== MaxLikeInf Results ===")
    print(f"SD: {result.SD:.2f}")
    print(f"TS: {result.TS:.3f}")
    print(f"slog: {slog:.4f}")
    print(f"ND: {result.ND:.0f}")
    print(f"k_1: {result.k_1:.3f}")
    print(f"Status: {status_message}")
    print(f"Iterations: {len(analyzer.optimization_steps)}")
    
    # Display SN curve
    display_sn_curve(fatigue_data, result, "MaxLikeInf")
    
    # Display convergence plot
    display_convergence_plot(analyzer.optimization_steps, "MaxLikeInf")
    
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
        'optimization_steps': analyzer.optimization_steps
    }
    
    return results_dict

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
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=['Likelihood', 'SD', 'TS', 'SD vs TS'])
    
    # Likelihood convergence
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['Likelihood'],
                            mode='lines+markers', name='Likelihood'),
                  row=1, col=1)
    
    # SD convergence  
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['SD'],
                            mode='lines+markers', name='SD'),
                  row=1, col=2)
    
    # TS convergence
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['TS'],
                            mode='lines+markers', name='TS'),
                  row=2, col=1)
    
    # Parameter space exploration
    fig.add_trace(go.Scatter(x=df_steps['SD'], y=df_steps['TS'],
                            mode='lines+markers', name='Path'),
                  row=2, col=2)
    
    fig.update_layout(title=f'{method_name} Convergence', height=600, showlegend=False)
    fig.show()


# %% Run MaxLikeInf analysis
if fatigue_data is not None:
    maxlike_results = run_maxlike_analysis(fatigue_data)
else:
    print("Cannot run analysis: no data loaded")
# %%
