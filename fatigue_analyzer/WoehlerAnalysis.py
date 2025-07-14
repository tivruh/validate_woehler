# -*- coding: utf-8 -*-
# ****************************************************************************
# * (C) 2025 Your Name/Schaeffler                                           *
# ****************************************************************************

__author__ = "Your Name, Schaeffler Technologies"
__credits__ = ["Your Name"]
__copyright__ = "(C) 2025 Your Name"
__license__ = ""
__version__ = "1.0"
__status__ = "Production"
__contact__ = "your.email@schaeffler.com"

from smatapps.core.core import SMatApp
import streamlit as st
import os
from ui_components import render_main, render_sidebar, apply_custom_styles, optimization_status, y_axis_controls
from results import display_results
from plots import PlotFatigue
from preprocess import load_and_prepare_data
from utils import LognormalAnalyzer

class WoehlerAnalysisApp(SMatApp):
    """ Fatigue Analyzer app for Wöhler curve analysis using PyLife """

    def __init__(self):
        """ Initialize the application with some general attributes and the standard layout """
        
        self.appname = "WoehlerAnalysis"
        super().__init__(self.appname)
        self.path = os.path.split(os.path.realpath(__file__))[0]
        self.url = self.get_baseUrl()
        self.init_design(self.appname)
        
    def build_gui(self):
        """ Build the graphical user interface of the actual app """
        
        apply_custom_styles()
        
        # STEP 1: Handle file upload and series selection
        uploaded_file, series_data, selected_series = render_main()
        
        if not uploaded_file or not series_data or not selected_series:
            st.info("Please upload an Excel file to start the analysis.")
            return
        
        # STEP 2: Process data (follow app.py)
        selected_data = {}
        detected_ngs = []
        any_survivors = False
        n_runouts = {}

        print("\nDebug: Processing selected series...")
        for series_name in selected_series:
            if series_name in series_data:
                series_info = series_data[series_name]
                df = series_info['data']
                print(f"Debug: Processing {series_name}, type: {type(df)}")
                
                # Store DataFrame directly (like original)
                selected_data[series_name] = {'data': df}
                
                # Auto-detect NG from censor column
                survivors = df[df['censor'] == 0] if 'censor' in df.columns else pd.DataFrame()
                if not survivors.empty:
                    ng_raw = int(survivors['cycles'].max())
                    ng = (ng_raw // 1000) * 1000
                    detected_ngs.append(ng)
                    any_survivors = True
                    print(f"Debug: {series_name} NG detected: {ng}")
                else:
                    ng = 5000000  # default
                    detected_ngs.append(ng)
                    print(f"Debug: {series_name} using default NG: {ng}")

        print(f"Debug: detected_ngs = {detected_ngs}")
        
        # STEP 3: Get UI parameters
        N_LCF, Ch1, load_type, curve_type, (lower_prob, upper_prob) = render_sidebar(any_survivors, n_runouts)
        
        # STEP 4: Create buttons and wait for clicks (like original)
        NG = max(detected_ngs) if detected_ngs else 5000000
        plotter = PlotFatigue(NG, Ch1, load_type, lower_prob, upper_prob, N_LCF)
        
        col1, col2 = st.columns(2)
        with col1:
            generate_full = st.button("Generate SN Curve")
        with col2:
            generate_endurance = st.button("Compare Endurance Limits")
        
        # STEP 5: Handle button clicks (like original)
        if generate_endurance:
            try:
                fig, results = plotter.create_endurance_comparison(selected_data)
                optimization_status(results)  # Now we have results!
                st.plotly_chart(fig, use_container_width=True)
                y_axis_controls()
                display_results(results, Ch1, any_survivors)
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    # st.set_page_config(page_title="Fatigue Analyser", layout="wide")
    app = WoehlerAnalysisApp()
    app.build_gui()