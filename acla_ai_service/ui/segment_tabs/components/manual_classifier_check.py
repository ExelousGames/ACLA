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
                        probs = segment_classifier.predict_segment_probabilities(snippet)
                        
                        st.write("Confidence per Label:")
                        # Filter and display
                        has_results = False
                        for label, score in probs.items():
                            if score > 0.01:
                                has_results = True
                                c_lab, c_prog = st.columns([1, 2])
                                with c_lab:
                                    label_str = LABEL_MAPPING.get(label, str(label))
                                    st.caption(f"{label_str} ({score:.1%})")
                                with c_prog:
                                    st.progress(score)
                        
                        if not has_results:
                            st.info("No labels detected with significant probability (>1%)")
                            
                    except Exception as e:
                        st.error(f"Error calling classifier: {str(e)}")
        else:
            st.info("Select a valid range (min length 1) to check probabilities.")
