# Backend

import math

import numpy as np

import plotly.graph_objects as go
import pylife
from pylife.materialdata import woehler

from scipy import optimize, stats
from plotly.subplots import make_subplots
from scipy.stats import norm

print(f"Using pylife from: {pylife.__file__}")
print(dir(pylife))
print(dir(pylife.materialdata))

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
        analyzer = HybridMaxLikeInf(fatigue_data)
        result = analyzer.analyze()
        
        # Add optimization status to the result
        if hasattr(analyzer, 'optimization_failed') and analyzer.optimization_failed:
            result.optimization_failed = analyzer.optimization_failed
            result.failure_reason = analyzer.failure_reason
        
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


class HybridMaxLikeInf(pylife.materialdata.woehler.MaxLikeInf):
    def __init__(self, fatigue_data, file_path=None):
        super().__init__(fatigue_data)
        self.file_path = file_path
    
    def _specific_analysis(self, wc):
        # Calculate initial parameters
        slope, lg_intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(self._fd.fractures.load),
            np.log10(self._fd.fractures.cycles)
        )
        
        # Initialize optimization
        SD_start = self._fd.fatigue_limit
        TS_start = 1.2
        
        # run Nelder-Mead 
        nm_result = optimize.fmin(lambda p: -self._lh.likelihood_infinite(p[0], p[1]),
                                [SD_start, TS_start], 
                                disp=False, full_output=True)
        
        # Extract results from Nelder-Mead
        SD = nm_result[0][0]
        TS = nm_result[0][1]
        
        # Get load range for validation
        min_load = self._fd.load.min()
        max_load = self._fd.load.max()
        
        self.optimization_failed = False
        self.failure_reason = None
        
        # Check for extreme values
        if SD < min_load * 0.5 or SD > max_load * 2.0:
            self.optimization_failed = True
            self.failure_reason = f"Optimization produced extreme SD value ({SD:.2f}) outside reasonable range [{min_load * 0.5:.2f}, {max_load * 1.5:.2f}]"
        
        # # If values are extreme, switch to L-BFGS-B with bounds
        # if SD < 50 or TS > 3.0:
        #     bounds = [(50, 20000), (1.0, 10.0)]
        #     lbfgs_result = optimize.minimize(
        #         lambda p: -self._lh.likelihood_infinite(p[0], p[1]),
        #         [SD_start, TS_start],
        #         method='L-BFGS-B',
        #         bounds=bounds
        #     )
        #     SD = lbfgs_result.x[0]
        #     TS = lbfgs_result.x[1]
        
        # Calculate ND and return results
        ND = 10**(lg_intercept + slope * (np.log10(SD)))
        
        wc['SD'] = SD
        wc['TS'] = TS
        wc['ND'] = ND
        wc['k_1'] = -slope
        
        return wc


class FatigueAnalyzer:
    def __init__(self, N_LCF, NG, Ch1, load_type, prob_levels=(0.05, 0.95)):
        self.N_LCF = N_LCF
        self.NG = NG
        self.Ch1 = Ch1
        self.load_type = load_type
        self.lower_prob, self.upper_prob = prob_levels
        
        print(f"Debug: User selected Probability Bands: {prob_levels}")
    
    
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
        
        
        # Check if optimization failed
        if hasattr(maxlike, 'optimization_failed') and maxlike.optimization_failed:
            print(f"Debug: Optimization failed: {maxlike.failure_reason}")
            return {
                'has_survivors': has_survivors,
                'n_runout': n_runout,
                'optimization_failed': True,
                'failure_reason': maxlike.failure_reason
            }
        
        SD = round(maxlike.SD, 2)  # Rounded to 2 decimal places for consistency
        k1 = round(maxlike.k_1, 1)
        ND = round(maxlike.ND)
        TN = round(maxlike.TN, 2)
        TS = round(maxlike.TS, 2)
        
        L_LCF = FatigueSolver.load_LCF(k1, ND or self.NG, SD, self.N_LCF)
        
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
            'optimization_failed': False
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


    def create_plot(self, series_data, curve_type="Full"):
        ranges = self.get_data_ranges(series_data)
        print("Using ranges for plot configuration:", ranges)

        fig = make_subplots()
        results = []
        
        colors = ['#648fff', '#fe6100', '#dc267f', '#785ef0', '#ffb000', '#000000']
        
        any_survivors = False

        for i, (series_name, series_info) in enumerate(series_data.items()):
            color = colors[i % len(colors)]
            # Process the DataFrame from the series info dictionary
            series_result = self.process_data(series_info['data'])
            
            # Check if optimization failed
            if series_result.get('optimization_failed', False):
                st.error(f"Optimization failed for series '{series_name}': {series_result.get('failure_reason', 'Unknown reason')}")
                st.warning("Please try a different dataset or contact support for assistance.")
                continue  # Skip further processing for this series
            
            series_result['series_name'] = series_name
            series_result['show_prob_lines'] = series_info['show_prob_lines']
            series_result['prob_levels'] = {
                'lower': self.lower_prob,
                'upper': self.upper_prob
            }
                
            self._plot_data(
                fig, series_info['data'], series_result, series_name, 
                color, curve_type)
            results.append(series_result)
            
            if series_result['has_survivors']:
                any_survivors = True
        
        self._format_plot(fig, any_survivors, ranges)
        fig.update_layout(
            title='Wöhler Curve'
        )
        return fig, results
    
    
    def get_data_ranges(self, series_data):
        """Get the min/max ranges for stress and cycles across all datasets"""
        min_stress = float('inf')
        max_stress = float('-inf')
        min_cycles = float('inf')
        max_cycles = float('-inf')
        
        for series_name, series_info in series_data.items():
            df = series_info['data']
            current_min_stress = df['load'].min()
            current_max_stress = df['load'].max()
            current_min_cycles = df['cycles'].min()
            current_max_cycles = df['cycles'].max()
            
            print(f"\nRanges for {series_name}:")
            print(f"Stress: {current_min_stress:.1f} to {current_max_stress:.1f}")
            print(f"Cycles: {current_min_cycles:.1f} to {current_max_cycles:.1f}")
            
            min_stress = min(min_stress, current_min_stress)
            max_stress = max(max_stress, current_max_stress)
            min_cycles = min(min_cycles, current_min_cycles)
            max_cycles = max(max_cycles, current_max_cycles)
        
        print(f"\nOverall ranges:")
        print(f"Stress: {min_stress:.1f} to {max_stress:.1f}")
        print(f"Cycles: {min_cycles:.1f} to {max_cycles:.1f}")
        
        return {
            'stress': {'min': min_stress, 'max': max_stress},
            'cycles': {'min': min_cycles, 'max': max_cycles}
        }
    
    
    def create_endurance_comparison(self, series_data):
        """Creates focused view of endurance limits with probability bands"""
        ranges = self.get_data_ranges(series_data)
        print("Using ranges for endurance view:", ranges)
        
        fig = make_subplots()
        results = []
        
        colors = ['#648fff', '#fe6100', '#dc267f', '#785ef0', '#ffb000', '#000000']
        any_survivors = False
        
        # Process each dataset
        for i, (series_name, series_info) in enumerate(series_data.items()):
            color = colors[i % len(colors)]
            
            # Process the data and collect results
            series_result = self.process_data(series_info['data'])
            series_result['series_name'] = series_name
            series_result['show_prob_lines'] = series_info['show_prob_lines']
            series_result['prob_levels'] = {
                'lower': self.lower_prob,
                'upper': self.upper_prob
            }
            
            if series_result['has_survivors']:
                any_survivors = True
                
                # Plot only runout region datapoints
                df = series_info['data']
                failures = df[df['failure']]
                survivors = df[~df['failure']]
                
                # Plot points after ND
                failures_hcf = failures[failures['cycles'] >= series_result['ND']]
                survivors_hcf = survivors[survivors['cycles'] >= series_result['ND']]
                
                if not failures_hcf.empty:
                    fig.add_trace(go.Scatter(
                        x=failures_hcf['cycles'], y=failures_hcf['load'],
                        mode='markers', marker=dict(color=color, symbol='cross'),
                        name=f'{series_name} (Failures)',
                        hovertemplate=f'<b>{series_name}</b><br>Cycles: <b>%{{x:.1f}}</b><br>Load: <b>%{{y}}</b><extra></extra>',
                        hoverlabel=dict(font=dict(color=color))
                    ))

                if not survivors_hcf.empty:
                    fig.add_trace(go.Scatter(
                        x=survivors_hcf['cycles'], y=survivors_hcf['load'],
                        mode='markers', marker=dict(color=color, symbol='triangle-right'),
                        name=f'{series_name} (Survivors)',
                        hovertemplate=f'<b>{series_name}</b><br>Cycles: <b>%{{y:.1f}}</b><extra></extra>',
                        hoverlabel=dict(font=dict(color=color))
                    ))
                
                # Plot endurance limit line (Pü50)
                fig.add_trace(go.Scatter(
                    x=[series_result['ND'], self.NG],
                    y=[series_result['SD'], series_result['SD']],
                    mode='lines',
                    line=dict(color=color),
                    name=f'{series_name} Pü50',
                    hovertemplate=f'<b>{series_name}</b><br>Pü50: <b>%{{y:.2f}}</b><extra></extra>',
                    hoverlabel=dict(font=dict(color=color))
                ))
                
                # Add probability bands if enabled
                if series_result.get('show_prob_lines', False):
                    slog = np.log10(series_result['TS'])/2.56
                    survival_probs = FatigueSolver.calculate_survival_probabilities(
                        series_result['SD'], 
                        slog,
                        self.lower_prob,
                        self.upper_prob
                    )
                    
                    for band_type, prob_value in [('lower', self.lower_prob), ('upper', self.upper_prob)]:
                        prob_key = f'PÜ{int(prob_value*100)}'
                        stress_value = survival_probs[prob_key]
                        
                        fig.add_trace(go.Scatter(
                            x=[series_result['ND'], self.NG],
                            y=[stress_value, stress_value],
                            mode='lines',
                            line=dict(color=color, dash='dot'),
                            name=f'{series_name} {prob_key}',
                            hovertemplate=f'<b>{series_name}</b><br>{prob_key}: <b>%{{y:.2f}}</b><extra></extra>',
                            hoverlabel=dict(font=dict(color=color))
                        ))
            
            results.append(series_result)
        
        # Get the minimum ND value from all series that have survivors
        min_nd = min((res['ND'] for res in results if res['has_survivors']), default=self.NG/10)
        
        self._format_plot(fig, any_survivors, ranges, endurance_view=True)
        
        # Update layout with specific title and adjust x-axis range
        fig.update_layout(
            title='Endurance Limit Comparison'
        )
        fig.update_xaxes(range=[math.log10(min_nd * 0.95), math.log10(self.NG * 1.05)])
        
        return fig, results
        
    
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
    
    
    
    
    def _plot_data(self, fig, df, results, series_name, color, curve_type):
        failures = df[df['failure']]
        survivors = df[~df['failure']]
        
        # Always plot data points
        if not failures.empty:
            fig.add_trace(go.Scatter(
                x=failures['cycles'], y=failures['load'],
                mode='markers', marker=dict(color=color, symbol='cross'),
                name=f'{series_name} (Failures)',
                hovertemplate=f'<b>{series_name}</b><br>Cycles: <b>%{{x:.1f}}</b><br>Load: <b>%{{y}}</b><br>Status: <b>Failure</b><extra></extra>',
                hoverlabel=dict(font=dict(color=color))
            ))

        if not survivors.empty:
            fig.add_trace(go.Scatter(
                x=survivors['cycles'], y=survivors['load'],
                mode='markers', marker=dict(color=color, symbol='triangle-right'),
                name=f'{series_name} (Survivors)',
                hovertemplate=f'<b>{series_name}</b><br>Cycles: <b>%{{x:.1f}}</b><br>Load: <b>%{{y}}</b><br>Status: <b>Survivor</b><extra></extra>',
                hoverlabel=dict(font=dict(color=color))
            ))

        if results['has_survivors']:
            # Calculate survival probabilities if needed
            survival_probs = None
            if results.get('show_prob_lines', False):
                slog = np.log10(results['TS'])/2.5361
                survival_probs = FatigueSolver.calculate_survival_probabilities(
                    results['SD'], 
                    slog,
                    self.lower_prob,
                    self.upper_prob
                )

            if curve_type == 'HCF':
                # Plot only HCF region (horizontal lines)
                # Main curve (Pü50)
                fig.add_trace(go.Scatter(
                    x=[results['ND'], self.NG],
                    y=[results['SD'], results['SD']],
                    mode='lines',
                    line=dict(color=color),
                    name=f'{series_name} HCF',
                    hovertemplate=f'<b>{series_name}</b><br>Pü50: <b>%{{y:.2f}}</b><extra></extra>',
                    hoverlabel=dict(font=dict(color=color)),
                    showlegend=False
                ))

                # Add probability bands if enabled
                if survival_probs and results.get('show_prob_lines', False):
                    for band_type, prob_value in [('lower', self.lower_prob), ('upper', self.upper_prob)]:
                        prob_key = f'PÜ{int(prob_value*100)}'
                        stress_value = survival_probs[prob_key]
                        
                        fig.add_trace(go.Scatter(
                            x=[results['ND'], self.NG],
                            y=[stress_value, stress_value],
                            mode='lines',
                            line=dict(color=color, dash='dot'),
                            name=f'{series_name} {prob_key}',
                            hovertemplate=f'<b>{series_name}</b><br>{prob_key}: <b>%{{y:.2f}}</b><extra></extra>',
                            hoverlabel=dict(font=dict(color=color)),
                            showlegend=False
                        ))
                        
            elif curve_type == 'Full':
                # Calculate starting point for Pü50 curve
                min_cycles = df['cycles'].min()
                L_LCF = FatigueSolver.load_LCF(results['k1'], results['ND'], results['SD'], min_cycles)
                
                # Plot main curve (Pü50)
                fig.add_trace(go.Scatter(
                    x=[min_cycles, results['ND'], self.NG],
                    y=[L_LCF, results['SD'], results['SD']],
                    mode='lines',
                    line=dict(color=color),
                    name=f'{series_name}',
                    hovertemplate=f'<b>{series_name}</b><br>Pü50: <b>%{{y:.2f}}</b><extra></extra>',
                    hoverlabel=dict(font=dict(color=color)),
                    showlegend=False
                ))

                # Add probability bands if enabled
                if survival_probs and results.get('show_prob_lines', False):
                    for band_type, prob_value in [('lower', self.lower_prob), ('upper', self.upper_prob)]:
                        prob_key = f'PÜ{int(prob_value*100)}'
                        stress_value = survival_probs[prob_key]
                        
                        # Calculate LCF starting point for this probability band
                        L_LCF_band = FatigueSolver.load_LCF(results['k1'], results['ND'], stress_value, min_cycles)
                        
                        fig.add_trace(go.Scatter(
                            x=[min_cycles, results['ND'], self.NG],
                            y=[L_LCF_band, stress_value, stress_value],
                            mode='lines',
                            line=dict(color=color, dash='dot'),
                            name=f'{series_name} {prob_key}',
                            hovertemplate=f'<b>{series_name}</b><br>{prob_key}: <b>%{{y:.2f}}</b><extra></extra>',
                            hoverlabel=dict(font=dict(color=color)),
                            showlegend=False
                        ))
    
    
    
    def _format_plot(self, fig, any_survivors, ranges, endurance_view=False):
        aspect_ratio = 1.3
        plot_width = 1000
        plot_height = plot_width / aspect_ratio
        
        fig.update_layout(
            autosize=False,
            width=plot_width,
            height=plot_height,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
                ),
            hovermode='closest',
            spikedistance=-1,
            hoverdistance=100
            
        )
        
        fig.update_xaxes(
            type="log", 
            title_text='Cycles', 
            showgrid=True,          
            gridwidth=1,             
            gridcolor='lightgray',   
            showspikes=True,
            spikemode='across',
            spikesnap='data',
            showline=True,
            spikecolor='gray',
            spikedash='dot',
            spikethickness=1,
            ticks="inside",
            ticklen=8,
            tickcolor='gray',
            # # tickformat=".0f", 
            # exponentformat="E",  
            minor=dict(
                tickmode='array',
                ticklen=4,
                tickvals=[j * 10**i for i in range(-3, 6) for j in range(2, 10)],
                gridcolor='lightgray',
                gridwidth=0.5,
                ticks="inside",
                tickcolor="gray",
                showgrid=True
            ),
            dtick='D1'
        )
        
        fig.update_yaxes(
            type="log", 
            title_text=f'{self.load_type} in {self.Ch1}', 
            showspikes=True,
            spikemode='across',
            spikesnap='data',
            showline=True,
            spikecolor='gray',
            spikedash='dot',
            spikethickness=1,
            ticks="inside",
            ticklen=8,
            tickcolor='gray',
            gridcolor='lightgray',
            # # tickformat=".0f",
            # exponentformat="E",
            minor=dict(
                ticklen=4,
                tickvals=[j * 10**i for i in range(-3, 6) for j in range(2, 10)],
                gridcolor='lightgray',
                gridwidth=0.5,
                ticks="inside",
                tickcolor="gray"
            ),
            dtick='D1',  # show all digits between powers of 10
        )
        
        if endurance_view:
            # Y-axis stays the same for high load ranges
            y_minor = dict(
                tickmode='array',
                ticklen=4,
                tickvals=[j * 10**i for i in range(2, 5) for j in range(1, 10)],
                gridcolor='lightgray',
                gridwidth=0.5,
                ticks="inside",
                tickcolor="gray",
                showgrid=True
            )
            
            # X-axis with 0.2M steps from 0 to 10M
            x_minor = dict(
                tickmode='array',
                ticklen=4,
                tickvals=[j * 2e5 for j in range(0, 51)],  # 0 to 10M in 0.2M steps
                gridcolor='lightgray',
                gridwidth=0.5,
                ticks="inside",
                tickcolor="gray",
                showgrid=True
            )

            fig.update_xaxes(
                minor=x_minor,
                tickmode='array',
                tickvals=[j * 2e5 for j in range(0, 51)],
                ticktext=[f"{j/5:.1f}M" for j in range(0, 51)]
            )
            
            fig.update_yaxes(
                minor=y_minor,
                type="log"
            )


#frontend_utils.py

import streamlit as st
import pandas as pd
import math

from io import BytesIO

# # Validation table formatting Settings 
# VALIDATION_THRESHOLD_LOW = 5
# VALIDATION_THRESHOLD_HIGH = 10



def render_main():
    # Homepage nav
    st.markdown(
        '<a href="https://materials-dev.schaeffler.com/" style="color: #00893d; font-size: 18px; font-weight: bold; text-decoration: none;">⬅ Return to MaterialsDevApps</a>',
        unsafe_allow_html=True
        )
    
    st.title("Fatigue Analyser")
    
    st.write("")
    
    col1, col2 = st.columns([3,2], gap="medium")
    with col2:
        try:
            pylife_version = pylife.__version__
        except AttributeError:
            pylife_version = "unknown (no __version__ attribute)"
            
        st.write(f"The evaluation is based on pylife v{pylife_version} or the maximum likelihood method." 
                "\n\nSupport: M. Funk (product owner), V. Arunachalam (optimizer calibration), M. Tikadar (app development)")
    
    with col1:
        # File uploader for multiple series
        uploaded_file = st.file_uploader("**Upload Excel file with results...**", type="xlsx")
        
        st.write("**The data must be uploaded in the correct format**")

        example_file = get_example_dataset()
        st.download_button(
            label="Download Example",
            data=example_file,
            file_name="fatigue_data_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    st.write("")
    
    # Dictionary to store dataframes and series names
    series_data = {}
    selected_series = []
    col1, col2 = None, None
    
    # template_file = get_excel_template()
    # st.download_button(
    #     label="Download Excel Template",
    #     data=template_file,
    #     file_name="fatigue_data_template.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    # )

    print("Debug: Before file processing")  # Debug print

    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        st.write("")
        st.write("")
            
        st.subheader("Select Series for Analysis")
        st.markdown(''':green-background[Please ensure that selected datasets have **the same Cycles to Runout**]''')
        st.write("")
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        print(f"Debug: Number of sheets: {len(sheet_names)}")  # Debug print
        
        for i, sheet in enumerate(sheet_names):
            with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
                df = pd.read_excel(xls, sheet_name=sheet)
                
                series_name = st.text_input(
                    f"Name for {sheet}", sheet, key=f"name_{sheet}")
                
                col_a, col_b = st.columns(2)
                
                with col_a:                
                    # Series inclusion and naming
                    include_series = st.checkbox(
                        f"Include {sheet}", value=True, key=f"include_{sheet}")
                
                with col_b:
                    # checkbox for probability lines
                    show_prob_lines = st.checkbox(
                    "Show bands", value=False, 
                    key=f"prob_lines_{sheet}",
                    help="Display probability lines showing scatter of endurance limit")

                if include_series:
                    series_data[series_name] = {
                        'data': df,
                        'show_prob_lines': show_prob_lines
                    }
                    selected_series.append(series_name)
                
                st.write("")
                    
    # print(f"Debug: series_data: {series_data}")  # Debug print
    # print(f"Debug: selected_series: {selected_series}")  # Debug print
    # print(f"Debug: col3 exists: {'col3' in locals()}")  # Debug print
    
    return uploaded_file, series_data, selected_series #, col3



def render_sidebar(any_survivors, n_runouts):
    st.sidebar.title("Input Parameters")
    
    default_n_runout = max(n_runouts.values()) if n_runouts else 10000000
    NG = st.sidebar.number_input(
        "Cycles to Runout:", value=int(default_n_runout), min_value=100000, step=100000)
    
    N_LCF = st.sidebar.number_input(
        "Pivot point in LCF:", value=10000, min_value=1000, step=1000)
    
    curve_options = ["Full", "LCF", "HCF"] if any_survivors else ["LCF"]
    print(f"Debug: Curve options: {curve_options}")
    
    # curve_type = st.sidebar.selectbox("Curve type:", curve_options)
    
    curve_type = "Full"
    
    st.write("")
    
    st.sidebar.subheader("Axis labels")
    load_type = st.sidebar.selectbox("Load type:", ["Amplitude", "Lower load", "Upper load", "Double amplitude"])
    Ch1 = st.sidebar.selectbox("Unit:", ["N", "mm", "Nm", "MPa", "°"])

    
    st.write("")
    
    # probability band configuration
    st.sidebar.subheader("Probability Bands")
    prob_options = {
        "Pü1/99": (0.01, 0.99),
        "Pü5/95": (0.05, 0.95),
        "Pü10/90": (0.10, 0.90)
    }
    selected_prob = st.sidebar.selectbox(
        "Select probability band:", 
        list(prob_options.keys()),
        index=1,  # Default to Pü5/95
        help="Select the probability levels for the scatter bands"
    )
    
    # Get the selected probability values
    lower_prob, upper_prob = prob_options[selected_prob]
    
    return N_LCF, NG, Ch1, load_type, curve_type, (lower_prob, upper_prob)



# def get_excel_template():
#     df_template = pd.DataFrame({
#         'load': [250, 250, 200, 200, 150, 150, 150],
#         'cycles': [100000, 150000, 500000, 2000000, 5000000, 5000000, 1800000],
#         'censor': [1, 1, 1, 1, 0, 0, 1]  # 0=survivor, 1=failure
#     })
#     buffer = BytesIO()
#     with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
#         df_template.to_excel(writer, sheet_name='Data1', index=False)
#         # Optional: Add a second sheet with different pattern
#         df_template.to_excel(writer, sheet_name='Data2', index=False)
#     return buffer.getvalue()


def get_example_dataset():
    """Create example Excel file using real fatigue test datasets"""
    df1 = pd.DataFrame({
        'load': [430, 342, 251, 184, 171, 158, 158, 158, 146, 135, 146, 135, 146, 251, 251, 251, 
                158, 146, 158, 171, 158, 171, 135, 135, 135, 135, 171, 171, 171],
        'cycles': [40867, 26765, 149829, 662852, 690450, 948124, 5000231, 1481447, 3917467, 
                5000256, 4071536, 5000256, 5000246, 113252, 271862, 173947, 4367292, 5000246, 
                5000231, 2505258, 669003, 1438884, 5000261, 5000216, 5000256, 5000251, 1167847, 
                1715098, 3953018],
        'censor': [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    })

    df2 = pd.DataFrame({
        'load': [300, 278, 257, 238, 220, 238, 257, 278, 257, 238, 220, 238, 257, 238, 220, 
                204, 340, 340, 340, 340, 340],
        'cycles': [415473, 1432994, 1514023, 1123808, 5000588, 5000551, 5000518, 4468138, 
                2627999, 4012433, 5000591, 5000563, 1368757, 2547519, 2456062, 5000585, 
                507702, 510192, 416487, 742479, 783257],
        'censor': [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    })

    df3 = pd.DataFrame({
        'load': [300, 350, 514, 441, 378, 350, 378, 408, 441, 408, 378, 408, 378, 378, 350, 324],
        'cycles': [5000200, 5000215, 126647, 609606, 1655278, 5000199, 5000196, 5000191, 
                1317151, 1128458, 5000205, 612549, 3459283, 3778766, 1736330, 5000196],
        'censor': [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    })

    # Create Excel file in memory
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Material_A', index=False)
        df2.to_excel(writer, sheet_name='Material_B', index=False)
        df3.to_excel(writer, sheet_name='Material_C', index=False)

    return buffer.getvalue()


def load_excel_data(file):
    try:
        df1 = pd.read_excel(file, sheet_name="Data1")
        df2 = pd.read_excel(file, sheet_name="Data2")
        df1.columns = ['load', 'cycles']
        df2.columns = ['load', 'cycles']
        return df1, df2
    except ValueError:
        df = pd.read_excel(file)
        df.columns = ['load', 'cycles']
        return df, None



def TS_to_slog(TS):
    return round(math.log10(TS) / 2.5361, 4)



def display_results(results, Ch1, any_survivors):
    data = []
    for res in results:
        if res is not None:
            # Get the probability levels from the results
            lower_prob = res['prob_levels']['lower']
            upper_prob = res['prob_levels']['upper']
            
            # Calculate the probability values using the configured levels
            slog = TS_to_slog(res['TS'])
            survival_probs = FatigueSolver.calculate_survival_probabilities(
                res['SD'], 
                slog,
                lower_prob,
                upper_prob
            )
            
            # Generate column names based on probability levels
            lower_col = f"PÜ{int(lower_prob*100)}"
            upper_col = f"PÜ{int(upper_prob*100)}"
            
            result_dict = {
                "Series": res['series_name'],                
                f"{lower_col} ({Ch1})": survival_probs[lower_col] if survival_probs else "N/A",
                f"PÜ50 ({Ch1})": round(res['SD'], 2),
                f"{upper_col} ({Ch1})": survival_probs[upper_col] if survival_probs else "N/A",
                "k": round(res['k1'], 4),
                "slog": slog,
                "ND": int(res['ND']) if res['ND'] is not None else "N/A"
            }
            data.append(result_dict)
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, 
                    hide_index=True,
                    column_config={
                        col: st.column_config.Column(
                            width="small"
                        ) for col in df.columns
                    }
        )
        
        # Update abbreviation meanings to include dynamic probability values
        st.markdown(f"""
        **Abbreviations:**
        - {lower_col}: Probability of Survival at {int(lower_prob*100)}%
        - PÜ50: Probability of Survival at 50%
        - {upper_col}: Probability of Survival at {int(upper_prob*100)}%
        - k: Slope of the S-N curve / Neigung der Wöhlerlinie
        - slog: Scatter of stress in log / Streuung der Spannung in log
        - ND: Knee point or Number of runouts / Kniepoint oder Nummer der Durchläufer
        """)
        
    else:
        st.warning("No valid results to display.")
    
    if not any_survivors:
        st.warning("No survivors detected in the data. Analysis is limited to LCF regime.")



# def validation(results):
#     st.subheader("Validation")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown(''':green-background[Enter Reference Pü50 Values]''')

#         validation_data = {}
        
#         for res in results:
#             series_name = res['series_name']
#             st.markdown(f"**{series_name}**")
#             validation_data[series_name] = {
#                 "Pü50": st.number_input(
#                     f"Pü50 for {series_name}", 
#                     value=399.0, 
#                     step=0.1,
#                     key=f"pu50_{series_name}"
#                 ),
#                 "slog": st.number_input(
#                     f"slog for {series_name}", 
#                     value=0.05, 
#                     step=0.001,
#                     key=f"slog_{series_name}"
#                 ),
#                 "k": st.number_input(
#                     f"k for {series_name}", 
#                     value=5.0, 
#                     step=0.1,
#                     key=f"k_{series_name}"
#                 )
#             }
#             st.markdown("---")
        
#         validate_button = st.button("Validate Results")
    
#     with col2:
#         if validate_button:
#             display_validation(results, validation_data)
#         else:
#             st.write("*Validation table appears here*")



# def display_validation(results, validation_data):
#     comparison_data = []
#     for res in results:
#         series_name = res['series_name']
#         calculated_pu50 = res['SD']
#         calculated_slog = TS_to_slog(res['TS'])
#         calculated_k = res['k1']
        
#         ref_data = validation_data.get(series_name, {})
#         reference_pu50 = ref_data.get('Pü50', 0)
#         reference_slog = ref_data.get('slog', 0)
#         reference_k = ref_data.get('k', 0)
        
#         # Calculate differences
#         pu50_difference = ((calculated_pu50 - reference_pu50) / reference_pu50 * 100) if reference_pu50 > 0 else "N/A"
#         slog_difference = ((calculated_slog - reference_slog) / reference_slog * 100) if reference_slog > 0 else "N/A"
#         k_difference = ((calculated_k - reference_k) / reference_k * 100) if reference_k > 0 else "N/A"
        
#         comparison_data.append({
#             "Series": series_name,
#             "Calc. Pü50": calculated_pu50,
#             "Ref. Pü50": reference_pu50,
#             "Pü50 Diff (%)": pu50_difference,
#             "Calc. slog": calculated_slog,
#             "Ref. slog": reference_slog,
#             "slog Diff (%)": slog_difference,
#             "Calc. k": calculated_k,
#             "Ref. k": reference_k,
#             "k Diff (%)": k_difference
#         })

#     comparison_df = pd.DataFrame(comparison_data)
    
#     styled_df = comparison_df.style.map(
#         apply_color_formatting, subset=['Difference (%)', 'Series'])
    
#     styled_df = styled_df.format({
#         'Calc. Pü50': '{:.2f}',
#         'Ref. Pü50': '{:.2f}',
#         'Pü50 Diff (%)': lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x,
#         'Calc. slog': '{:.4f}',
#         'Ref. slog': '{:.4f}',
#         'slog Diff (%)': lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x,
#         'Calc. k': '{:.2f}',
#         'Ref. k': '{:.2f}',
#         'k Diff (%)': lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x
#     })    
    
#     # Convert the styled Dataframe to HTML
#     styled_html = styled_df.hide().to_html()
    
#     st.markdown(styled_html, unsafe_allow_html=True)


# def apply_color_formatting(value, name=None):
#     if isinstance(value, str):
#         return ''
    
#     # Check if this is a difference column
#     if 'Diff (%)' in name:
#         if abs(value) <= VALIDATION_THRESHOLD_LOW:
#             return 'background-color: #90EE90'  # light green
#         elif abs(value) <= VALIDATION_THRESHOLD_HIGH:
#             return 'background-color: #FFDE59'  # light yellow
#         else:
#             return 'background-color: #FFB6C1'  # light red
#     return ''



# def display_validation(results, validation_data):
#     comparison_data = []
#     for res in results:
#         series_name = res['series_name']
#         calculated_pu50 = res['SD']
#         calculated_slog = TS_to_slog(res['TS'])
#         calculated_k = res['k1']
        
#         ref_data = validation_data.get(series_name, {})
#         reference_pu50 = ref_data.get('Pü50', 0)
#         reference_slog = ref_data.get('slog', 0)
#         reference_k = ref_data.get('k', 0)
        
#         # Calculate differences
#         pu50_difference = ((calculated_pu50 - reference_pu50) / reference_pu50 * 100) if reference_pu50 > 0 else "N/A"
#         slog_difference = ((calculated_slog - reference_slog) / reference_slog * 100) if reference_slog > 0 else "N/A"
#         k_difference = ((calculated_k - reference_k) / reference_k * 100) if reference_k > 0 else "N/A"
        
#         comparison_data.append({
#             "Series": series_name,
#             "Calc. Pü50": calculated_pu50,
#             "Ref. Pü50": reference_pu50,
#             "Pü50 Diff (%)": pu50_difference,
#             "Calc. slog": calculated_slog,
#             "Ref. slog": reference_slog,
#             "slog Diff (%)": slog_difference,
#             "Calc. k": calculated_k,
#             "Ref. k": reference_k,
#             "k Diff (%)": k_difference
#         })

#     comparison_df = pd.DataFrame(comparison_data)
    
#     # Apply styling
#     styled_df = comparison_df.style.apply(
#         apply_color_formatting,
#         axis=0  # Apply function to each column
#     )
    
#     styled_df = styled_df.format({
#         'Calc. Pü50': '{:.2f}',
#         'Ref. Pü50': '{:.2f}',
#         'Pü50 Diff (%)': lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x,
#         'Calc. slog': '{:.4f}',
#         'Ref. slog': '{:.4f}',
#         'slog Diff (%)': lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x,
#         'Calc. k': '{:.2f}',
#         'Ref. k': '{:.2f}',
#         'k Diff (%)': lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x
#     })    
    
#     # Convert the styled Dataframe to HTML
#     styled_html = styled_df.hide().to_html()
    
#     st.markdown(styled_html, unsafe_allow_html=True)
    


# styles.py

import streamlit as st

# Custom CSS for styling
def apply_custom_styles():
    page_title="Wöhler Fatigue Analyser"

    # Custom CSS for styling
    st.markdown("""
        <style>
        .logo-container {
            position: fixed;
            right: 40px;
            top: 40px;
            z-index: 999; 
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(12, 149, 76, 0.2);
        }
        .logo-container img {
            width: 250px;
        }

        .download-button {
            display: flex;
            justify-content: space-around;
            align items: center;
            padding: 1em 0;
            width: 100%
            margin: 0 0.5em
        }
        
        .download-button .stButton > button {
            width: 100%
            margin: 0 0.5em
        }
        
        .stButton > button {
            color: #0C954C !important;
            border-width: 0.5px !important;
            border-style: solid !important;
        }
        
        .section-spacing {
            margin-top: 3rem;
        }
        
        /* Align checkboxes vertically */
        .stCheckbox {
            padding-top: 0.5rem;
        }
        
        .stTable {
            padding: 0.5em;
            border: 0.05em solid #e6e6e6;
            border-radius: 0.25em;
        }
        
        </style>
        <div class="logo-container">
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/72/Schaeffler_logo.svg" alt="Schaeffler Logo">
        </div>
        """, unsafe_allow_html=True)



# Autor: Matthias Funk
# Short description: Fatigue Analysis Tool

import streamlit as st


st.set_page_config(page_title="Fatigue Analyser", layout="wide")


def main():
    apply_custom_styles()
    
    print("Debug: Starting main function")
    # uploaded_file, series_data, selected_series, runout_column = render_main()
    uploaded_file, series_data, selected_series = render_main()
    
    if uploaded_file is not None and series_data and selected_series:
        selected_data = {name: series_data[name] for name in selected_series}
        
        # Process data to check for survivors first
        temp_analyzer = FatigueAnalyzer(10000, 10000000, "N", "Amplitude")
        any_survivors, n_runouts = temp_analyzer.get_runouts(selected_data)
        
        # Display runouts in the 3rd column
        # with runout_column:
        #     st.subheader("Runout Cycles")
        #     for series, runout in n_runouts.items():
        #         st.write(f"{series}: {runout:,} cycles")        
        
        N_LCF, NG, Ch1, load_type, curve_type, (lower_prob, upper_prob) = render_sidebar(any_survivors, n_runouts)
        analyzer = FatigueAnalyzer(N_LCF, NG, Ch1, load_type, prob_levels=(lower_prob, upper_prob))
        
        st.write("")
        st.write("")

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            generate_full = st.button("Generate Wöhler Curve")
        with col2:
            generate_endurance = st.button("Compare Endurance Limits")
            
        st.write("")
        st.write("")
                
        if generate_full:
            fig, results = analyzer.create_plot(selected_data, "Full")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("")
            st.subheader("Analysis Results")
            display_results(results, Ch1, any_survivors)

        if generate_endurance:
            fig, results = analyzer.create_endurance_comparison(selected_data)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("")
            st.subheader("Analysis Results")
            display_results(results, Ch1, any_survivors)
        
        # if results:
        #     validation(results)
        
    else:
        st.info("Please upload an Excel file to start the analysis.")


if __name__ == "__main__":
    print("Debug: Calling main function")
    main()
