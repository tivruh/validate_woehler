import streamlit as st
import pandas as pd
import math
from utils import FatigueSolver

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
                # "k": round(res['k1'], 4),
                "slog": slog
                # "ND": int(res['ND']) if res['ND'] is not None else "N/A"
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
        - slog: Scatter of stress in log / Streuung der Spannung in log
        """)
        # - k: Slope of the S-N curve / Neigung der Wöhlerlinie
        # - ND: Knee point or Number of runouts / Kniepoint oder Nummer der Durchläufer
        
    else:
        st.warning("No valid results to display.")
    
    if not any_survivors:
        st.warning("No survivors detected in the data. Analysis is limited to LCF regime.")


def TS_to_slog(TS):
    return round(math.log10(TS) / 2.5361, 4)