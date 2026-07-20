import streamlit as st
import pandas as pd

def render_classifier_probability_check(df: pd.DataFrame, form_start: float, form_end: float, LABEL_MAPPING: dict):
    with st.expander("Classifier Probabilities (AI Check)"):
        if form_start < form_end and int(form_end) < len(df):
            if st.button("Check Probabilities for Range", key="manual_check_probs_btn"):
                with st.spinner("Analyzing segment with Classifier..."):
                    try:
                        # Import here to avoid circular dependencies during initial load
                        from app.ml.segment_classifier.service import segment_classifier
                        
                        # Extract segment
                        snippet = df.iloc[int(form_start):int(form_end)]
                        scores = segment_classifier.score_sequence(snippet)
                        visible = [
                            label for label in scores.columns
                            if float(scores[label].max()) > 0.01
                        ]
                        if visible:
                            display_scores = scores[visible].rename(columns={
                                label: LABEL_MAPPING.get(label, label)
                                for label in visible
                            })
                            st.line_chart(display_scores)
                        else:
                            st.info("No temporal label score exceeded 1%.")
                            
                    except Exception as e:
                        st.error(f"Error calling classifier: {str(e)}")
        else:
            st.info("Select a valid range (min length 1) to check probabilities.")
