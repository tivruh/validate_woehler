import streamlit as st
from ui_components import render_main, render_sidebar, apply_custom_styles, optimization_status
from results import display_results
from plots import PlotFatigue
from preprocess import load_and_prepare_data
from utils import LognormalAnalyzer

st.set_page_config(page_title="Fatigue Analyser", layout="wide")

def main():
    # Apply CSS styling first
    apply_custom_styles()
    
    # STEP 1: Handle file upload and series selection
    # This gives us the raw data structure with user preferences
    uploaded_file, series_data, selected_series = render_main()
    
    # Exit early if no file uploaded - clean separation of concerns
    if not uploaded_file or not series_data or not selected_series:
        st.info("Please upload an Excel file to start the analysis.")
        return
    
    # STEP 2: Process all data once - centralized data processing
    # This eliminates duplicate processing and provides consistent results
    print("Debug: Starting centralized data processing...")
    
    # Initialize containers for processed results
    selected_data = {name: series_data[name] for name in selected_series}
    detected_ngs = []
    n_runouts = {}
    any_survivors = False
    processing_errors = []
    
    # Process each selected series once and store all results
    for series_name, series_info in selected_data.items():
        try:
            # Auto-detect NG from censor column (Feature 2: no user NG input needed)
            
            fatigue_data, detected_ng, sd_bounds, df_prepared = load_and_prepare_data(series_info['data'])
            
            if fatigue_data is None:
                processing_errors.append(f"Failed to process {series_name}")
                continue
                
            detected_ngs.append(detected_ng)
            
            # Calculate runout info for sidebar display
            survivors = df_prepared[~df_prepared['fracture']]
            n_runout = survivors['cycles'].min() if not survivors.empty else None
            if n_runout:
                n_runout = round(n_runout / 10000) * 10000
                any_survivors = True
            n_runouts[series_name] = n_runout
            
            # Run fatigue analysis using our optimized LognormalAnalyzer
            analyzer = LognormalAnalyzer(fatigue_data)
            result = analyzer.analyze()
            
            # Store complete results for plotting and display
            # This structure matches what PlotFatigue expects
            series_info['processed_result'] = {
                'SD': result.SD,
                'TS': result.TS, 
                'ND': result.ND,
                'k1': result.k_1,  # Note: PyLife uses k_1, plotting expects k1
                'TN': result.TN,
                'has_survivors': not survivors.empty,
                'series_name': series_name,
                'optimization_success': getattr(result, 'optimization_success', True),
                'optimization_message': getattr(result, 'optimization_message', 'Success'),
                'optimization_iterations': getattr(result, 'optimization_iterations', 0)
            }
            
        except Exception as e:
            processing_errors.append(f"Error processing {series_name}: {str(e)}")
            print(f"Error processing {series_name}: {e}")
    
    # Show processing errors if any occurred
    if processing_errors:
        for error in processing_errors:
            st.error(error)
    
    # Exit if no successful processing
    if not detected_ngs:
        st.error("No datasets could be processed successfully.")
        return
    
    # STEP 3: Get user interface parameters
    # Now we have real data to inform the sidebar
    N_LCF, Ch1, load_type, curve_type, (lower_prob, upper_prob) = render_sidebar(any_survivors, n_runouts)
    
    # Use maximum detected NG across all datasets
    # This ensures compatibility when multiple datasets have different runout cycles
    NG = max(detected_ngs)
    print(f"Debug: Using NG = {NG} (max of detected values: {detected_ngs})")
    
    # STEP 4: Create plotting interface
    # Direct instantiation - no wrapper classes needed
    plotter = PlotFatigue(NG, Ch1, load_type, lower_prob, upper_prob, N_LCF)
    
    # UI layout for plot generation buttons
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        generate_full = st.button("Generate Wöhler Curve")
    with col2:
        generate_endurance = st.button("Compare Endurance Limits")
    
    st.write("")
    st.write("")
    
    # STEP 5: Handle plot generation requests
    # Clean separation - plotting only happens when requested
    
    if generate_full:
        try:
            # Generate full Wöhler curve analysis
            fig, results = plotter.create_plot(selected_data, "Full")
            
            # Display optimization status in collapsible section (existing feature)
            optimization_status(results)
            
            # Display plot and results
            st.plotly_chart(fig, use_container_width=True)
            st.write("")
            st.subheader("Analysis Results")
            display_results(results, Ch1, any_survivors)
            
        except Exception as e:
            st.error(f"Error generating Wöhler curve: {str(e)}")
            print(f"Plotting error: {e}")
    
    if generate_endurance:
        try:
            # Generate endurance limit comparison
            fig, results = plotter.create_endurance_comparison(selected_data)
            
            # Display optimization status
            optimization_status(results)
            
            # Display plot and results
            st.plotly_chart(fig, use_container_width=True)
            st.write("")
            st.subheader("Analysis Results")
            display_results(results, Ch1, any_survivors)
            
        except Exception as e:
            st.error(f"Error generating endurance comparison: {str(e)}")
            print(f"Plotting error: {e}")

if __name__ == "__main__":
    main()