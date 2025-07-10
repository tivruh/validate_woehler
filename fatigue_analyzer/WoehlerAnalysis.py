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
        
        # Apply CSS styling first
        apply_custom_styles()
        
        # STEP 1: Handle file upload and series selection
        uploaded_file, series_data, selected_series = render_main()
        
        # Exit early if no file uploaded
        if not uploaded_file or not series_data or not selected_series:
            st.info("Please upload an Excel file to start the analysis.")
            return
        
        # STEP 2: Sidebar configuration  
        with st.sidebar:
            analysis_settings = render_sidebar(series_data, selected_series)
            
        # STEP 3: Process data for selected series
        processed_data = {}
        all_series_valid = True
        
        for series_name in selected_series:
            if series_name in series_data:
                self.log.info(f"Processing series: {series_name}")
                result = load_and_prepare_data(
                    series_data[series_name], 
                    analysis_settings['ng'], 
                    analysis_settings.get('n_lcf', 10000)
                )
                
                if result:
                    processed_data[series_name] = result
                else:
                    all_series_valid = False
                    st.error(f"Failed to process data for series: {series_name}")
        
        if not all_series_valid or not processed_data:
            st.error("Some series could not be processed. Please check your data format.")
            return
            
        # STEP 4: Run analysis and display results
        st.subheader("Analysis Results")
        
        # Display optimization status
        optimization_status()
        
        # Y-axis controls
        y_axis_controls()
        
        # Run analysis and display results
        display_results(processed_data, analysis_settings)
        
        self.log.info("Analysis completed successfully")

if __name__ == '__main__':
    st.set_page_config(page_title="Fatigue Analyser", layout="wide")
    app = WoehlerAnalysisApp()
    app.build_gui()