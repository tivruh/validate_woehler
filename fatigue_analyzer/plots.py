import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


class PlotFatigue:
    def __init__(self, NG, Ch1, load_type, lower_prob, upper_prob, N_LCF):
        self.N_LCF = N_LCF
        self.NG = NG
        self.Ch1 = Ch1
        self.load_type = load_type
        self.lower_prob = lower_prob
        self.upper_prob = upper_prob

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
            series_result = series_info['processed_result']
            
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
        
        self._format_plot(fig)
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
            series_result = series_info['processed_result']
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
        
        self._format_plot(fig, endurance_view=True)
        
        # Update layout with specific title and adjust x-axis range
        fig.update_layout(
            title='Endurance Limit Comparison'
        )
        fig.update_xaxes(range=[math.log10(min_nd * 0.95), math.log10(self.NG * 1.05)])
        
        return fig, results    
    
    
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
    
    
    def _format_plot(self, fig, endurance_view=False):
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

