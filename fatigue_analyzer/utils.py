import math
import numpy as np
from scipy.stats import norm
import pylife
from scipy import optimize, stats
from pylife.materialdata import woehler


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
                f'PÜ{lower_prob*100:g}': round(PÜ_lower, 2),
                'PÜ50': round(PÜ50, 2),
                f'PÜ{upper_prob*100:g}': round(PÜ_upper, 2)
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
