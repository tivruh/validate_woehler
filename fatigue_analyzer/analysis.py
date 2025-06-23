import numpy as np
import pylife.materialdata.woehler as woehler
import math
from scipy import optimize, stats
from scipy.stats import norm
import pylife
from plots import PlotFatigue

class FatigueSolver:
    @staticmethod
    def scatter(PÜ, PÜ50, slog):     
        q = norm.ppf(1-PÜ, loc=0, scale=1)
        return 10**(math.log10(PÜ50)+slog*q)
    
    
    @staticmethod
    def load_LCF(k, N, S, N1):
        # return 10**(math.log10(S)-(math.log10(N/N1))/-k)
        try:
            # Check for invalid inputs
            if k == 0 or N <= 0 or S <= 0 or N1 <= 0:
                print(f"Invalid input in load_LCF. k={k}, N={N}, S={S}, and N1={N1}")
                return None
            
            # Calculate the result
            result = 10**(math.log10(S)-(math.log10(N/N1))/-k)
            
            # Check if the result is valid
            if math.isnan(result) or math.isinf(result):
                print(f"Warning: load_LCF Calculation resulted in an invalid value. Inputs: k={k}, N={N}, S={S}, and N1={N1}")
                return None
            
            return result
        except Exception as e:
            print(f"Error in load_LCF calculation: {str(e)}")
            print(f"Inputs: k={k}, N={N}, S={S}, N1={N1}")
            # Return a default value or handle the error as appropriate for your application
            return None  # or some other appropriate default value
        
    
    @staticmethod
    def maxl(df, NG):
        df = woehler.determine_fractures(df, NG)
        fatigue_data = df.fatigue_data
        analyzer = LognormalAnalyzer(fatigue_data)
        result = analyzer.analyze()
        
        # Add optimization status to the result
        result.optimization_success = analyzer.optimization_success
        result.optimization_message = analyzer.optimization_message
        result.optimization_iterations = analyzer.optimization_iterations
        
        return result
    
    
    @staticmethod
    def calculate_survival_probabilities(PÜ50, slog, lower_prob=0.05, upper_prob=0.95):
        """Calculate fatigue strength values for 5%, 50%, and 95% survival probabilities
        
        This function calculates the stress levels corresponding to different survival probabilities
    using the scatter factor (slog) and the median strength value (PÜ50).
    
    Args:
        PÜ50 (float): The median fatigue strength (50% survival probability)
        slog (float): The scatter factor in logarithmic form
        lower_prob (float): The lower probability level (e.g., 0.05 for Pü5)
        upper_prob (float): The upper probability level (e.g., 0.95 for Pü95)
    
    Returns:
        dict: A dictionary containing the stress values for lower, median, and upper probabilities
        """
        try:
            PÜ_lower = FatigueSolver.scatter(lower_prob, PÜ50, slog)
            PÜ_upper = FatigueSolver.scatter(upper_prob, PÜ50, slog)
            
            return {
                f'PÜ{int(lower_prob*100)}': round(PÜ_lower, 2),
                'PÜ50': round(PÜ50, 2),
                f'PÜ{int(upper_prob*100)}': round(PÜ_upper, 2)
            }
        except Exception as e:
            print(f"Error calculating survival probabilities: {e}")
            return None



class LognormalAnalyzer(pylife.materialdata.woehler.Elementary):
    def __init__(self, fatigue_data, file_path=None):
        super().__init__(fatigue_data)
        self.file_path = file_path
        self.optimization_success = False
        self.optimization_message = ""
        self.optimization_iterations = 0
    
    def _specific_analysis(self, wc):
        """
        Direct lognormal optimization approach - replaces PyLife's likelihood calculation
        """
        # Get original dataframe
        df = self._fd._obj
        
        # Separate failures and runouts using fracture field (not censor)
        failure_data = df[df['fracture'] == True]
        failure = failure_data['load'].tolist()
        
        runout_data = df[df['fracture'] == False]
        runout = runout_data['load'].tolist()
        
        # Calculate initial values
        if len(failure) > 0 and len(runout) > 0:
            max_val = np.log(max(failure))
            min_val = np.log(min(runout))
            avg = (max_val + min_val) / 2
        else:
            # Fallback if either group is empty
            all_loads = df['load'].tolist()
            max_val = np.log(max(all_loads))
            min_val = np.log(min(all_loads))
            avg = (max_val + min_val) / 2
        
        # Define the negative log-likelihood function
        def negative_log_likelihood(params):
            mu, sigma = params
            
            if len(failure) > 0 and len(runout) > 0:
                nll = -(np.sum(np.log(stats.lognorm.cdf(failure, s=sigma, loc=0, scale=np.exp(mu)))) + 
                        np.sum(np.log(1 - stats.lognorm.cdf(runout, s=sigma, loc=0, scale=np.exp(mu)))))
            elif len(failure) > 0:
                nll = -np.sum(np.log(stats.lognorm.cdf(failure, s=sigma, loc=0, scale=np.exp(mu))))
            elif len(runout) > 0:
                nll = -np.sum(np.log(1 - stats.lognorm.cdf(runout, s=sigma, loc=0, scale=np.exp(mu))))
            else:
                return np.inf
            
            return nll
        
        # Initial parameters
        initial_params = [avg, 1]  # Initial guess for mu (log of scale) and sigma
        
        # Optimize parameters
        result = optimize.minimize(
            negative_log_likelihood, 
            initial_params,
            method='Nelder-Mead',
            options={'disp': False}
        )
        
        # Store optimization results
        self.optimization_success = result.success
        self.optimization_message = result.message
        self.optimization_iterations = result.nit if hasattr(result, 'nit') else 0
        
        # Extract optimized parameters
        optimized_mu, optimized_sigma_ln = result.x
        optimized_sigma = 0.43429448 * optimized_sigma_ln  # Convert to base-10 log
        
        # Convert to PyLife parameters
        final_sd = np.exp(optimized_mu)  # Scale parameter
        final_ts = np.exp(2.5631 * optimized_sigma)  # Convert to TS format
        
        # Calculate ND using optimized SD (consistent with other methods)
        slope = wc['k_1']
        lg_intercept = np.log10(wc['ND']) - (-slope) * np.log10(wc['SD'])
        final_nd = 10**(lg_intercept + (-slope) * np.log10(final_sd))
        
        # Update wc with optimized values
        wc['SD'] = final_sd
        wc['TS'] = final_ts
        wc['ND'] = final_nd
        
        return wc


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
    
