# huck.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylife.utils.functions import std_to_scattering_range
import pylife.materialdata.woehler as woehler

class HuckMethod(woehler.Elementary):
    """
    Implementation of Huck's method for fatigue strength analysis
    as described in DIN standards for staircase testing.
    """
    
    def __init__(self, fatigue_data):
        super().__init__(fatigue_data)
        
    def _specific_analysis(self, wc):
        """
        Implement Huck's method for calculating fatigue strength
        and scatter parameters from staircase test data.
        """
        print("\n=== Huck's Method Analysis ===")
        
        # Extract load levels and sort them
        loads = sorted(self._fd.load.unique())
        
        # Identify the lowest load level to be included (L0)
        L0 = min(loads)
        
        # Assign ordinal numbers to load levels (i values)
        load_to_i = {load: i for i, load in enumerate(loads)}
        
        # Calculate the increment (d_log) between load levels
        if len(loads) > 1:
            # Using geometric mean of ratios for consistent increment
            ratios = [loads[i+1]/loads[i] for i in range(len(loads)-1)]
            d_log = np.exp(np.mean(np.log(ratios)))
            print(f"Load increment (d_log): {d_log:.4f}")
        else:
            d_log = 1.0
            print("Warning: Only one load level found, using d_log = 1.0")
        
        # Group data and calculate fi, i·fi, i²·fi
        result_table = []
        for load in loads:
            i = load_to_i[load]
            # Count failures at this load level
            failures_at_load = self._fd.fractures[self._fd.fractures.load == load].shape[0]
            
            if failures_at_load > 0:  # Only include levels with failures
                result_table.append({
                    'i': i,
                    'Load': load,
                    'fi': failures_at_load,
                    'i·fi': i * failures_at_load,
                    'i²·fi': i**2 * failures_at_load
                })
        
        # Convert to DataFrame for easier calculation and display
        df_results = pd.DataFrame(result_table)
        print("\nStaircase Analysis Table:")
        print(df_results)
        
        # Calculate characteristic numbers
        F_T = df_results['fi'].sum()         # Total of fi values
        A_T = df_results['i·fi'].sum()       # Total i·fi
        B_T = df_results['i²·fi'].sum()      # Total i²·fi
        
        print(f"\nCharacteristic Numbers:")
        print(f"F_T (Σfi) = {F_T}")
        print(f"A_T (Σi·fi) = {A_T}")
        print(f"B_T (Σi²·fi) = {B_T}")
        
        if F_T == 0:
            raise ValueError("No failures detected in data, cannot apply Huck's method")
        
        # Calculate mean fatigue strength using equation (56)
        L_aLNG = L0 * (d_log ** (A_T / F_T))
        
        # Calculate variance using equation (57)
        D_T = (F_T * B_T - A_T**2) / (F_T**2)
        print(f"Variance (D_T): {D_T:.6f}")
        
        # Calculate standard deviation using equation (58) or (59)
        if D_T < 0.5:
            s_logL = 0.5 * np.log10(d_log)
            print(f"Using equation (58) for standard deviation (D_T < 0.5)")
        else:
            # Complex formula from equation (59)
            term1 = np.log10(d_log)
            term2 = 10**(4.579494 * np.power(F_T, -0.889521))
            term3 = np.power(D_T, 7.235548 * np.power(F_T, -0.405229))
            s_logL = term1 * term2 * term3
            print(f"Using equation (59) for standard deviation (D_T ≥ 0.5)")
        
        print(f"Standard deviation (s_logL): {s_logL:.6f}")
        
        # Calculate scatter in load direction (TS)
        TS = 10**(2.564 * s_logL)  # 10% to 90% probability (±1.28 sigma)
        
        # Convert to PyLife parameters
        wc['SD'] = L_aLNG  # Mean fatigue strength
        wc['TS'] = TS      # Scatter
        
        # For ND (knee point), we'll keep the ND from Elementary's common analysis
        # (already calculated in _common_analysis and stored in wc)
        
        # Print final results
        print("\nFinal Huck Parameters:")
        print(f"L_aLNG (mean fatigue strength): {L_aLNG:.2f}")
        print(f"TS (scatter): {TS:.4f}")
        print(f"s_logL (log standard deviation): {s_logL:.6f}")
        
        # Calculate 10% and 90% failure probabilities
        L_aLNG_10p = 10**(np.log10(L_aLNG) - 1.28 * s_logL)
        L_aLNG_90p = 10**(np.log10(L_aLNG) + 1.28 * s_logL)
        
        print(f"L_aLNG,10% (10% failure probability): {L_aLNG_10p:.2f}")
        print(f"L_aLNG,90% (90% failure probability): {L_aLNG_90p:.2f}")
        
        return wc
        
    def plot_staircase(self):
        """Create a visualization of the staircase data with results."""
        # Get sorted load levels
        loads = sorted(self._fd.load.unique())
        
        # Get results
        L_aLNG = self._obj.SD  # Mean fatigue strength from analysis
        s_logL = np.log10(self._obj.TS) / 2.56  # Standard deviation
        
        # Calculate 10% and 90% failure probabilities
        L_aLNG_10p = 10**(np.log10(L_aLNG) - 1.28 * s_logL)
        L_aLNG_90p = 10**(np.log10(L_aLNG) + 1.28 * s_logL)
        
        # Initialize figure
        plt.figure(figsize=(10, 6))
        
        # Plot load levels
        for load in loads:
            plt.axhline(y=load, color='lightgray', linestyle='-', alpha=0.5)
        
        # Plot test points
        for i, row in self._fd._obj.iterrows():
            x = i
            y = row.load
            if row.fracture:
                plt.plot(x, y, 'ko', markersize=8)  # Black circle for fracture
            else:
                plt.plot(x, y, 'ko', markerfacecolor='white', markersize=8)  # White circle for runout
        
        # Plot fatigue strength lines
        x_range = plt.xlim()
        plt.hlines(L_aLNG, x_range[0], x_range[1], colors='blue', linestyles='-', label=f'Mean ({L_aLNG:.1f})')
        plt.hlines(L_aLNG_10p, x_range[0], x_range[1], colors='red', linestyles='--', label=f'10% ({L_aLNG_10p:.1f})')
        plt.hlines(L_aLNG_90p, x_range[0], x_range[1], colors='green', linestyles='--', label=f'90% ({L_aLNG_90p:.1f})')
        
        plt.title("Staircase Test Analysis (Huck's Method)")
        plt.xlabel("Test Number")
        plt.ylabel("Load")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()