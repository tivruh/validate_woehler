import streamlit as st
import pandas as pd
from io import BytesIO
import pylife

def render_main():
    # Homepage nav
    st.markdown(
        '<a href="https://materials-dev.schaeffler.com/" style="color: #00893d; font-size: 18px; font-weight: bold; text-decoration: none;">⬅ Return to MaterialsDevApps</a>',
        unsafe_allow_html=True
        )
    
    st.title("Fatigue Analyzer")
    
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
        "Pü2.5/97.5": (0.025, 0.975),
        "Pü10/90": (0.10, 0.90)
    }
    selected_prob = st.sidebar.selectbox(
        "Select probability band:", 
        list(prob_options.keys()),
        index=1,  # Default to Pü2.5/97.5
        help="Select the probability levels for the scatter bands"
    )
    
    # Get the selected probability values
    lower_prob, upper_prob = prob_options[selected_prob]
    
    return N_LCF, Ch1, load_type, curve_type, (lower_prob, upper_prob)



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


def optimization_status(results):
    """Display optimization status for all series in collapsible section"""
    with st.expander("⚙ Optimization Status", expanded=False):
        for result in results:
            if result:
                series_name = result.get('series_name', 'Unknown')
                opt_success = result.get('optimization_success', True)
                opt_message = result.get('optimization_message', 'Success')
                opt_iterations = result.get('optimization_iterations', 0)
                
                if opt_success:
                    st.success(f"**{series_name}**: ✅ {opt_message} ({opt_iterations} iterations)")
                else:
                    st.error(f"**{series_name}**: ❌ {opt_message} ({opt_iterations} iterations)")


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
