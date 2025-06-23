import numpy as np
import pylife.materialdata.woehler as woehler
import math
from scipy import optimize, stats
from scipy.stats import norm
from plots import PlotFatigue
from utils import FatigueSolver, LognormalAnalyzer


class ProcessData:
    def __init__(self, N_LCF, NG):
        self.N_LCF = N_LCF
        self.NG = NG

    
    def process_data(self, df):
        # Check if 'censor' column exists
        has_censor = 'censor' in df.columns
        
        if has_censor:
            # use censor to determine failures
            df['failure'] = df['censor'] == 1
            survivors = df[df['censor'] == 0]
        else:
            # if no censor, determine based on NG
            df['failure'] = df['cycles'] < self.NG
            survivors = df[~df['failure']]
        
        has_survivors = not df['failure'].all()
        print(f"Debug: Has survivors: {has_survivors}")
        
        # Calculate n_runout
        n_runout = survivors['cycles'].min() if not survivors.empty else None
        if n_runout is not None:
            n_runout = round(n_runout / 10000) * 10000
        
        print(f"Debug: Lowest survivor cycle: {n_runout}")
        
        if not has_survivors:
            return self.process_data_no_survivors(df)
        
        maxlike = FatigueSolver.maxl(df, self.NG)
        print(f"Debug: maxlike.SD = {maxlike.SD}, maxlike.k_1 = {maxlike.k_1}")

        # Get optimization status from new LognormalAnalyzer
        optimization_success = getattr(maxlike, 'optimization_success', True)
        optimization_message = getattr(maxlike, 'optimization_message', 'Success')
        optimization_iterations = getattr(maxlike, 'optimization_iterations', 0)

        print(f"Debug: Optimization - Success: {optimization_success}, Message: {optimization_message}")
        
        SD = round(maxlike.SD, 2)  # Rounded to 2 decimal places for consistency
        k1 = round(maxlike.k_1, 1)
        ND = round(maxlike.ND)
        TN = round(maxlike.TN, 2)
        TS = round(maxlike.TS, 2)
        
        L_LCF = FatigueSolver.load_LCF(k1, self.NG, SD, self.N_LCF)
        
        x_LCF = [self.N_LCF, ND, self.NG]
        y_LCF = [L_LCF, SD, SD]
        x_HCF = [ND, self.NG]
        y_HCF = [SD, SD]
        
        print(f"Debug: x_LCF = {x_LCF}, y_LCF = {y_LCF}")
        print(f"Debug: x_HCF = {x_HCF}, y_HCF = {y_HCF}")
        
        return {
            'SD': SD, 'k1': k1, 'ND': ND, 'TN': TN, 'TS': TS,
            'x_LCF': x_LCF, 'y_LCF': y_LCF,
            'x_HCF': x_HCF, 'y_HCF': y_HCF,
            'df': df, # Return the dataframe with failure information
            'has_survivors': has_survivors,
            'n_runout': n_runout,
            'optimization_failed': False,
            'optimization_success': optimization_success,
            'optimization_message': optimization_message,
            'optimization_iterations': optimization_iterations
        }
    
    
    def process_data_no_survivors(self, df):
        print("Debug: Processing data with no survivors")
        df['failure'] = True  # All data points are failures
        
        maxlike = FatigueSolver.maxl(df, self.NG)
        print(f"Debug: maxlike.SD = {maxlike.SD}, maxlike.k_1 = {maxlike.k_1}")
        
        SD = round(maxlike.SD, 2)
        k1 = round(maxlike.k_1, 1)
        TN = round(maxlike.TN, 2)
        TS = round(maxlike.TS, 2)
        
        L_LCF = FatigueSolver.load_LCF(k1, self.NG, SD, self.N_LCF)
        
        x_LCF = [self.N_LCF, self.NG]
        y_LCF = [L_LCF, SD]
        
        print(f"Debug: x_LCF = {x_LCF}, y_LCF = {y_LCF}")
        
        return {
            'SD': SD, 'k1': k1, 'ND': None, 'TN': TN, 'TS': TS,
            'x_LCF': x_LCF, 'y_LCF': y_LCF,
            'x_HCF': None, 'y_HCF': None,
            'df': df,
            'has_survivors': False,
            'n_runout': None
        }


    def get_runouts(self, series_data):
        """Process data to detect runouts and survivors"""
        n_runouts = {}
        any_survivors = False
        
        for series_name, series_info in series_data.items():
            df = series_info['data']  # Get the DataFrame from the dictionary
            series_result = self.process_data(df)
            n_runouts[series_name] = series_result['n_runout']
            if series_result['has_survivors']:
                any_survivors = True
                
        return any_survivors, n_runouts


class FatigueAnalyzer:
    def __init__(self, N_LCF, NG, Ch1, load_type, prob_levels=(0.025, 0.975)):
        self.N_LCF = N_LCF
        self.NG = NG
        self.Ch1 = Ch1
        self.load_type = load_type
        self.lower_prob, self.upper_prob = prob_levels
        
        # Create helper classes with their needed configuration
        self.data_processor = ProcessData(N_LCF, NG)
        self.plotter = PlotFatigue(NG, Ch1, load_type, self.lower_prob, self.upper_prob, N_LCF)

    def create_plot(self, series_data, curve_type="Full"):
        return self.plotter.create_plot(series_data, curve_type)

    def create_endurance_comparison(self, series_data):
        return self.plotter.create_endurance_comparison(series_data)

    def get_runouts(self, series_data):
        return self.data_processor.get_runouts(series_data)

    def _get_lcf_start_point(self, df, target_stress, k1, ND):
        """Calculate the starting point for LCF curves based on minimum cycles in data
        
        Args:
            df: DataFrame containing the test data
            target_stress: The stress level at the knee point (can be Pü50, Pü5, etc)
            k1: Slope of the curve
            ND: Knee point cycles
            
        Returns:
            tuple: (min_cycles, load_at_min) - The x,y coordinates where the curve should start
        """
        min_cycles = df['cycles'].min()
        
        # Using the same slope k1, calculate what the load should be at min_cycles
        L_LCF = FatigueSolver.load_LCF(k1, ND, target_stress, min_cycles)
        
        return min_cycles, L_LCF

    
    def _get_curve_coordinates(self, curve_type, min_cycles, ND, NG, start_value, end_value):
        """
        Get the appropriate coordinates for plotting based on curve type.
        
        Args:
            curve_type: The type of curve to plot ('Full', 'LCF', or 'HCF')
            min_cycles, ND, NG: The cycle values for curve segments
            start_value, end_value: The stress values for curve segments
        
        Returns:
            tuple: (x_coordinates, y_coordinates) for plotting
        """
        if curve_type == 'LCF':
            return [min_cycles, ND], [start_value, end_value]
        elif curve_type == 'HCF':
            return [ND, NG], [end_value, end_value]
        else:  # 'Full'
            return [min_cycles, ND, NG], [start_value, end_value, end_value]
    
