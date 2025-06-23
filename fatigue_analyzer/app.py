import streamlit as st
from ui_components import render_main, render_sidebar, apply_custom_styles
from results import display_results
from analysis import FatigueAnalyzer
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

        for series_name, series_info in selected_data.items():
            # Use new function that auto-detects NG from censor
            fatigue_data, _, _ = load_and_prepare_data(series_info['data'])
            # Create simple processed result
            series_info['processed_result'] = {
                'fatigue_data': fatigue_data,
                'has_survivors': True,  # will determine properly later
                'series_name': series_name
            }

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