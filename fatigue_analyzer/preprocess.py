import pandas as pd
import numpy as np
import pylife.materialdata.woehler as woehler
import traceback

def load_and_prepare_data(df_input):
    """Load fatigue data and auto-detect NG from censor column"""
    try:
        # Load data
        df_test = df_input.copy()
        
        # Rename 'loads' to 'load' if needed
        if 'loads' in df_test.columns:
            df_test = df_test.rename(columns={'loads': 'load'})
        
        # Auto-detect NG from survivors (censor = 0)
        survivors = df_test[df_test['censor'] == 0]
        if not survivors.empty:
            ng_raw = int(survivors['cycles'].max())  # Raw highest survivor cycle
            ng = (ng_raw // 1000) * 1000  # Round down to nearest 1000
            print(f"Auto-detected NG: {ng_raw:,} → {ng:,} cycles (rounded down to nearest 1000)")
        else:
            ng_raw = int(df_test['cycles'].max())  # Fallback: use max cycles if no survivors
            ng = (ng_raw // 1000) * 1000  # Round down to nearest 1000
            print(f"No survivors found. Using max cycles: {ng_raw:,} → {ng:,} cycles (rounded down to nearest 1000)")
        
        # Rest of function remains the same...
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

        return fatigue_data, ng, sd_bounds, df_prepared
        
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None, None, None