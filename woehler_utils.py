# woehler_utils.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import optimize
import traceback

def analyze_fatigue_file(file_path):
    """Basic function to load and inspect fatigue test data"""
    try:
        print(f"\nLoading file: {file_path}")
        
        # Read test data
        df_test = pd.read_excel(file_path, sheet_name='Data')
        
        # Rename to 'load' (if column name is 'loads')
        if 'loads' in df_test.columns:
            df_test = df_test.rename(columns={'loads': 'load'})
            
        # Read Jurojin reference values if they exist
        try:
            df_ref = pd.read_excel(file_path, sheet_name='Jurojin_results', header=None)
            
            # Extract reference values
            ref_values = {}
            for idx, row in df_ref.iterrows():
                param_name = row[0]
                param_value = row[1]
                if isinstance(param_value, str) and ',' in param_value:
                    param_value = float(param_value.replace(',', '.'))
                ref_values[param_name] = param_value
        except:
            ref_values = None
            print("No reference values found")
        
        return df_test, ref_values
        
    except Exception as e:
        print(f"Error processing {file_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        return None, None


def run_optimization_with_tracking(likelihood_obj, initial_values, method='nelder-mead', bounds=None):
    """Run optimization with tracking for either method and plot results
    
    Parameters:
    -----------
    likelihood_obj : Likelihood object
        The likelihood object from PyLife
    initial_values : list
        Initial parameter values [SD_start, TS_start]
    method : str
        'nelder-mead' or 'l-bfgs-b'
    bounds : list of tuples, optional
        Required for L-BFGS-B: [(SD_min, SD_max), (TS_min, TS_max)]
    """
    # Create list to store optimization steps
    optimization_steps = []
    
    # Define objective function with tracking
    def tracked_objective(p):
        # Calculate likelihood
        likelihood = likelihood_obj.likelihood_infinite(p[0], p[1])
        
        # Store current step
        optimization_steps.append({
            'Step': len(optimization_steps) + 1,
            'SD': p[0],
            'TS': p[1],
            'Likelihood': likelihood
        })
        
        # Return negative for minimization
        return -likelihood
    
    # Initial values
    SD_start, TS_start = initial_values
    print(f"Initial values - SD: {SD_start:.2f}, TS: {TS_start:.2f}")
    
    # Run appropriate optimizer
    if method.lower() == 'nelder-mead':
        result = optimize.fmin(
            tracked_objective,
            initial_values,
            disp=True,
            full_output=True
        )
        
        # Extract results
        SD, TS = result[0]
        warnflag = result[4]
        message = result[5] if len(result) > 5 else "No message"
        success = (warnflag == 0)
        
        # Map warnflag to meaning
        warnflag_meanings = {
            0: "Success - optimization converged",
            1: "Maximum number of iterations/evaluations reached",
            2: "Function values not changing (precision loss)",
            3: "NaN result encountered"
        }
        status_text = warnflag_meanings.get(warnflag, "Unknown")
        
    elif method.lower() == 'l-bfgs-b':
        if bounds is None:
            raise ValueError("Bounds required for L-BFGS-B method")
            
        result = optimize.minimize(
            tracked_objective,
            initial_values,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Extract results
        SD, TS = result.x
        success = result.success
        message = result.message
        status_text = "Success - optimization converged" if success else "Failed to converge"
    
    # Print results
    print(f"\n{method.upper()} optimization status: {status_text}")
    print(f"Message: {message}")
    print(f"Final values - SD: {SD:.2f}, TS: {TS:.2f}")
    
    # Calculate slog
    slog = np.log10(TS)/2.5361
    print(f"Calculated slog: {slog:.4f}")
    
    # Check if values are reasonable
    min_load = likelihood_obj._fd.load.min()
    max_load = likelihood_obj._fd.load.max()
    
    reasonable_values = True
    if SD < min_load * 0.5 or SD > max_load * 2.0:
        print(f"WARNING: SD value {SD:.2f} outside reasonable range [{min_load*0.5:.2f}, {max_load*2.0:.2f}]")
        reasonable_values = False
    
    if TS < 1.0 or TS > 10.0:
        print(f"WARNING: TS value {TS:.2f} outside typical range [1.0, 10.0]")
        reasonable_values = False
    
    print(f"Values reasonable: {reasonable_values}")
    
    # Plot convergence
    plot_optimization_convergence(optimization_steps, method)
    
    # Return results
    return {
        'method': method,
        'SD': SD, 
        'TS': TS,
        'success': success,
        'message': message,
        'reasonable_values': reasonable_values,
        'optimization_steps': optimization_steps
    }
    

def plot_optimization_convergence(steps, method="optimization"):
    """Plot optimization convergence"""
    # Convert to DataFrame
    df_steps = pd.DataFrame(steps)
    
    # Set proper method name for display
    if method.lower() == 'nelder-mead':
        display_method = 'Nelder-Mead'
    elif method.lower() == 'l-bfgs-b':
        display_method = 'L-BFGS-B'
    else:
        display_method = method.capitalize()
    
    # Create figure
    fig = go.Figure()
        
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['Likelihood'], 
                            mode='lines+markers', name='Likelihood'))
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['SD'], 
                            mode='lines+markers', name='SD'))
    fig.add_trace(go.Scatter(x=df_steps['Step'], y=df_steps['TS'], 
                            mode='lines+markers', name='TS'))
    
    fig.update_layout(
        title=f'{display_method.capitalize()} Convergence',
        xaxis_title='Step',
        yaxis_title='Value',
        width=800,
        height=600
    )

    fig.show()
    return fig

def create_comparison_plot(df_prepared, elementary_result, ml_result, huck_result, N_LCF=10000, NG=5000000, file_name=None):
    """
    Create an SN curve plot comparing Nelder-Mead and Huck's method
    with highlighted staircase region
    
    Parameters:
    -----------
    df_prepared : DataFrame
        Fatigue test data with 'load', 'cycles', 'censor', 'fracture' columns
    elementary_result : Series
        Elementary analysis result with ND value for identifying staircase region
    ml_result : Series
        MaxLikeInf (Nelder-Mead) analysis result
    huck_result : Series
        Huck's method analysis result
    N_LCF : int
        Pivot point in LCF
    NG : int
        Maximum number of cycles
    file_name : str
        Name of the file being analyzed for the title
    """
    # Create figure
    fig = make_subplots()
    
    # Get the staircase region boundary (Elementary ND)
    elementary_ND = elementary_result.ND
    
    # Separate points into different groups
    slope_failures = df_prepared[(df_prepared['fracture']) & (df_prepared['cycles'] < elementary_ND)]
    slope_survivors = df_prepared[(~df_prepared['fracture']) & (df_prepared['cycles'] < elementary_ND)]
    staircase_failures = df_prepared[(df_prepared['fracture']) & (df_prepared['cycles'] >= elementary_ND)]
    staircase_survivors = df_prepared[(~df_prepared['fracture']) & (df_prepared['cycles'] >= elementary_ND)]
    
    # Plot slope region points (standard blue)
    if not slope_failures.empty:
        fig.add_trace(go.Scatter(
            x=slope_failures['cycles'], y=slope_failures['load'],
            mode='markers', marker=dict(color='#648fff', symbol='cross', size=10),
            name='Slope Region Failures',
            hovertemplate='Cycles: %{x:.1f}<br>Load: %{y}<br>Status: Failure<extra></extra>'
        ))
    
    if not slope_survivors.empty:
        fig.add_trace(go.Scatter(
            x=slope_survivors['cycles'], y=slope_survivors['load'],
            mode='markers', marker=dict(color='#648fff', symbol='triangle-right', size=10),
            name='Slope Region Survivors',
            hovertemplate='Cycles: %{x:.1f}<br>Load: %{y}<br>Status: Survivor<extra></extra>'
        ))
    
    # Plot staircase region points (red for highlighting)
    if not staircase_failures.empty:
        fig.add_trace(go.Scatter(
            x=staircase_failures['cycles'], y=staircase_failures['load'],
            mode='markers', marker=dict(color='#dc267f', symbol='cross', size=10),
            name='Staircase Region Failures',
            hovertemplate='Cycles: %{x:.1f}<br>Load: %{y}<br>Status: Failure<extra></extra>'
        ))
    
    if not staircase_survivors.empty:
        fig.add_trace(go.Scatter(
            x=staircase_survivors['cycles'], y=staircase_survivors['load'],
            mode='markers', marker=dict(color='#dc267f', symbol='triangle-right', size=10),
            name='Staircase Region Survivors',
            hovertemplate='Cycles: %{x:.1f}<br>Load: %{y}<br>Status: Survivor<extra></extra>'
        ))
    
    # Plot Nelder-Mead curve
    k_nm = ml_result.k_1
    ND_nm = ml_result.ND
    SD_nm = ml_result.SD
    
    # Calculate LCF starting point for Nelder-Mead
    L_LCF_nm = 10**(np.log10(SD_nm)-(np.log10(ND_nm/N_LCF))/-k_nm)
    
    # Plot Nelder-Mead LCF curve
    fig.add_trace(go.Scatter(
        x=[N_LCF, ND_nm],
        y=[L_LCF_nm, SD_nm],
        mode='lines', line=dict(color='#648fff', width=2),
        name='Nelder-Mead (LCF)'
    ))
    
    # Plot Nelder-Mead HCF curve
    fig.add_trace(go.Scatter(
        x=[ND_nm, NG],
        y=[SD_nm, SD_nm],
        mode='lines', line=dict(color='#648fff', width=2, dash='dash'),
        name='Nelder-Mead (HCF)'
    ))
    
    # Plot Huck curve
    k_huck = huck_result.k_1  # Same as Elementary/NM since not recalculated
    ND_huck = huck_result.ND
    SD_huck = huck_result.SD
    
    # Calculate LCF starting point for Huck
    L_LCF_huck = 10**(np.log10(SD_huck)-(np.log10(ND_huck/N_LCF))/-k_huck)
    
    # Plot Huck LCF curve
    fig.add_trace(go.Scatter(
        x=[N_LCF, ND_huck],
        y=[L_LCF_huck, SD_huck],
        mode='lines', line=dict(color='#dc267f', width=2),
        name='Huck (LCF)'
    ))
    
    # Plot Huck HCF curve
    fig.add_trace(go.Scatter(
        x=[ND_huck, NG],
        y=[SD_huck, SD_huck],
        mode='lines', line=dict(color='#dc267f', width=2, dash='dash'),
        name='Huck (HCF)'
    ))
    
    # Add vertical line at Elementary ND to show staircase boundary
    fig.add_vline(
        x=elementary_ND,
        line=dict(color='green', width=2, dash='dot'),
        annotation_text="Staircase Boundary (ND)",
        annotation_position="top right"
    )
    
    # Add markers for both recalculated ND values
    fig.add_trace(go.Scatter(
        x=[ND_nm], 
        y=[SD_nm * 0.9],  # Just below the line for visibility
        mode='markers+text',
        marker=dict(color='#648fff', size=12, symbol='triangle-down'),
        text=["NM ND"],
        textposition="bottom center",
        name='Nelder-Mead ND',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[ND_huck], 
        y=[SD_huck * 0.9],  # Just below the line for visibility
        mode='markers+text',
        marker=dict(color='#dc267f', size=12, symbol='triangle-down'),
        text=["Huck ND"],
        textposition="bottom center",
        name='Huck ND',
        showlegend=False
    ))
    
    # Calculate slog values
    slog_nm = np.log10(ml_result.TS)/2.5361
    slog_huck = np.log10(huck_result.TS)/2.5361
    
    # Create title with parameters - with smaller font size for subtitle
    title_text = f"SN Curve Comparison"
    if file_name:
        title_text += f" - {file_name}"
        
    subtitle = (
        f"<span style='font-size:10px; color:#648fff'>Nelder-Mead: k={k_nm:.2f}, ND={int(ND_nm):,}, Pü50={SD_nm:.1f}, slog={slog_nm:.3f}</span><br>"
        f"<span style='font-size:10px; color:#dc267f'>Huck: k={k_huck:.2f}, ND={int(ND_huck):,}, Pü50={SD_huck:.1f}, slog={slog_huck:.3f}</span><br>"
        f"<span style='font-size:10px; color:green'>Staircase region begins at ND={int(elementary_ND):,} cycles</span>"
    )
    
    fig.update_layout(
        title={
            'text': title_text + "<br>" + subtitle,
            'y': 0.95,  # Move title up to add more space
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 14}  # Slightly smaller main title
        },
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Cycles",
        yaxis_title="Load",
        showlegend=True,
        width=1000,
        height=700,
        margin=dict(t=100),  # Add more top margin for the title
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig