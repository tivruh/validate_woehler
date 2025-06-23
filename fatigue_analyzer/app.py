import streamlit as st
from ui_components import render_main, render_sidebar, apply_custom_styles
from results import display_results
from analysis import FatigueAnalyzer
from utils import LognormalAnalyzer
from preprocess import load_and_prepare_data

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
        
        N_LCF, Ch1, load_type, curve_type, (lower_prob, upper_prob) = render_sidebar(any_survivors, n_runouts)
        
        st.write("")
        st.write("")

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            generate_full = st.button("Generate Wöhler Curve")
        with col2:
            generate_endurance = st.button("Compare Endurance Limits")
            
        st.write("")
        st.write("")

        detected_ngs = []
        for series_name, series_info in selected_data.items():
            # Load data with auto-detect NG
            fatigue_data, detected_ng, sd_bounds, df_prepared = load_and_prepare_data(series_info['data'])
            detected_ngs.append(detected_ng)
            
            # Run analysis to get SD, TS, ND, k1
            analyzer_temp = LognormalAnalyzer(fatigue_data)
            result = analyzer_temp.analyze()
            
            # Create processed_result with all needed parameters
            series_info['processed_result'] = {
                'SD': result.SD,
                'TS': result.TS, 
                'ND': result.ND,
                'k1': result.k_1,  # Note: k_1 vs k1
                'TN': result.TN,
                'has_survivors': True,
                'series_name': series_name
            }
        
        NG = max(detected_ngs)
        analyzer = FatigueAnalyzer(N_LCF, NG, Ch1, load_type, prob_levels=(lower_prob, upper_prob))

        if generate_full:
            fig, results = analyzer.create_plot(selected_data, "Full")
            
            # Display optimization status for each series
            with st.expander("⚙ Optimization Status", expanded=False):
                for result in results:
                    if result is not None:
                        series_name = result.get('series_name', 'Unknown')
                        opt_success = result.get('optimization_success', True)
                        opt_message = result.get('optimization_message', 'Success')
                        opt_iterations = result.get('optimization_iterations', 0)
                        
                        if opt_success:
                            st.success(f"**{series_name}**: ✅ {opt_message} ({opt_iterations} iterations)")
                        else:
                            st.error(f"**{series_name}**: ❌ {opt_message} ({opt_iterations} iterations)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("")
            st.subheader("Analysis Results")
            display_results(results, Ch1, any_survivors)

        if generate_endurance:
            fig, results = analyzer.create_endurance_comparison(selected_data)
            
            # Display optimization status for each series
            with st.expander("⚙ Optimization Status", expanded=False):
                for result in results:
                    if result is not None:
                        series_name = result.get('series_name', 'Unknown')
                        opt_success = result.get('optimization_success', True)
                        opt_message = result.get('optimization_message', 'Success')
                        opt_iterations = result.get('optimization_iterations', 0)
                        
                        if opt_success:
                            st.success(f"**{series_name}**: ✅ {opt_message} ({opt_iterations} iterations)")
                        else:
                            st.error(f"**{series_name}**: ❌ {opt_message} ({opt_iterations} iterations)")
            
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