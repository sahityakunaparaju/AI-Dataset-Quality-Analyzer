import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

from app.services.report_generator import generate_pdf_report
from app.services.drift_detection import detect_data_drift, compare_dataset_versions


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Dataset Quality Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------
# MODERN UI STYLE
# ---------------------------------------------------
st.markdown("""
<style>

/* overall spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* metric cards */
[data-testid="stMetric"] {
    border-radius: 10px;
    padding: 12px;
    background-color: rgba(240,242,246,0.6);
}

/* chart spacing */
.stPlotlyChart {
    padding-top: 10px;
}

/* sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(200,200,200,0.2);
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Dataset Quality Platform")

page = st.sidebar.selectbox(
    "Navigation",
    ["Analysis", "Dataset Comparison", "History"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Dataset Monitoring")


# ===================================================
# DATASET ANALYSIS
# ===================================================
if page == "Analysis":

    st.title("Dataset Quality Analysis")

    upload_col, target_col = st.columns([2,1])

    with upload_col:
        file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    with target_col:
        target = st.text_input("Target Column (optional)")


    if file is not None:
        st.session_state["file_bytes"] = file.getvalue()
        st.session_state["file_name"] = file.name


    # ------------------------------------------------
    # RUN ANALYSIS
    # ------------------------------------------------
    if st.button("Run Analysis") and file is not None:

        with st.spinner("Running dataset diagnostics..."):

            files = {
                "file": (
                    st.session_state["file_name"],
                    st.session_state["file_bytes"],
                    "text/csv"
                )
            }

            data = {"target_column": target} if target else {}

            try:

                resp = requests.post(
                    "http://127.0.0.1:8000/api/analyze",
                    files=files,
                    data=data,
                    timeout=120
                )

                if resp.status_code != 200:
                    st.error(resp.text)

                else:
                    result = resp.json()
                    st.session_state["analysis_result"] = result

            except Exception as e:
                st.error(str(e))


    # ------------------------------------------------
    # SHOW RESULTS
    # ------------------------------------------------
    if "analysis_result" in st.session_state:

        result = st.session_state["analysis_result"]
        fname = st.session_state.get("file_name","dataset")

        stats = result.get("statistics", {})
        score_obj = result.get("health_score", {})
        score_val = score_obj.get("score", 0)


        # ------------------------------------------------
        # HEALTH SCORE + OVERVIEW
        # ------------------------------------------------
        col1, col2 = st.columns([1,2])

        with col1:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val,
                title={"text":"Dataset Health Score"},
                gauge={
                    "axis": {"range":[0,100]},
                    "bar":{"color":"#2563EB"},
                    "steps":[
                        {"range":[0,40],"color":"#ef4444"},
                        {"range":[40,70],"color":"#f59e0b"},
                        {"range":[70,100],"color":"#10b981"}
                    ]
                }
            ))

            fig.update_layout(height=260)
            st.plotly_chart(fig, width="stretch")


        with col2:

            st.subheader("Dataset Overview")

            c1,c2,c3,c4 = st.columns(4)

            c1.metric("Rows", f"{stats.get('rows',0):,}")
            c2.metric("Columns", stats.get("columns",0))
            c3.metric("Numeric", stats.get("numeric_feature_count",0))
            c4.metric("Categorical", stats.get("categorical_feature_count",0))

        # ------------------------------------------------
        # RECOMMENDATIONS
        # ------------------------------------------------
        st.markdown("---")
        st.subheader("Data Quality Recommendations")

        recs = result.get("recommendations",{}).get("recommendations",[])

        if recs:
            for r in recs:
                st.warning(r)
        else:
            st.success("No major issues detected.")


        # ------------------------------------------------
        # REPORT DOWNLOAD
        # ------------------------------------------------
        st.markdown("---")
        st.subheader("Reports")

        json_str = json.dumps(result, indent=2)

        colA,colB = st.columns(2)

        colA.download_button(
            "Download JSON Report",
            data=json_str,
            file_name=f"{fname}_report.json",
            mime="application/json"
        )

        try:

            pdf_bytes = generate_pdf_report(result)

            colB.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"{fname}_report.pdf",
                mime="application/pdf"
            )

        except:
            st.warning("PDF generation unavailable.")


        # ------------------------------------------------
        # DATA PREVIEW
        # ------------------------------------------------
        st.markdown("---")
        st.subheader("Dataset Preview")

        try:

            preview_df = pd.read_csv(
                pd.io.common.BytesIO(st.session_state["file_bytes"])
            )

            st.write(
                f"Rows: {preview_df.shape[0]} | Columns: {preview_df.shape[1]}"
            )

            st.dataframe(preview_df.head(20), width="stretch")

        except:
            st.info("Preview unavailable.")


        # ------------------------------------------------
        # DIAGNOSTICS
        # ------------------------------------------------
        st.markdown("---")

        tab1,tab2,tab3,tab4,tab5 = st.tabs([
            "Missing Data",
            "Correlations",
            "Outliers",
            "ML Insights",
            "Leakage Detection"
        ])


        # Missing Data
        with tab1:

            mv = result.get("missing_values",{}).get("columns",[])

            if mv:

                df_mv = pd.DataFrame(mv)

                fig = px.bar(
                    df_mv,
                    x="column",
                    y="missing_pct",
                    color="risk_level"
                )

                st.plotly_chart(fig, width="stretch")


        # Correlations
        with tab2:

            corr = result.get("correlations",{})

            if corr.get("matrix"):

                df_corr = pd.DataFrame(corr["matrix"])

                fig = px.imshow(
                    df_corr,
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1
                )

                st.plotly_chart(fig, width="stretch")


        # Outliers
        with tab3:

            out = result.get("outliers",{})

            z_df = pd.DataFrame(out.get("zscore_per_feature",[]))

            if not z_df.empty:

                fig = px.bar(
                    z_df,
                    x="feature",
                    y="outliers"
                )

                st.plotly_chart(fig, width="stretch")


        # ML Insights
        with tab4:

            baseline = result.get("baseline_model")

            if baseline and "error" not in baseline:

                if baseline["model_type"] == "classification":

                    metric1,metric2,metric3,metric4 = st.columns(4)

                    metric1.metric("Accuracy", round(baseline.get("accuracy",0),3))
                    metric2.metric("Precision", round(baseline.get("precision",0),3))
                    metric3.metric("Recall", round(baseline.get("recall",0),3))
                    metric4.metric("F1 Score", round(baseline.get("f1_score",0),3))

                else:

                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        st.metric("RMSE", round(baseline.get("rmse",0),3))

                    with metric_col2:
                        st.metric("R² Score", round(baseline.get("r2_score",0),3)) 


                if "top_features" in baseline:

                    fi_df = pd.DataFrame(baseline["top_features"])

                    fig = px.bar(
                        fi_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        color="importance",
                        color_continuous_scale="Blues",
                        title="Top Predictive Features"
                    )
                    fig.update_layout(height=400)

                    st.plotly_chart(fig, width="stretch")

                else:
                    st.info("Provide a target column to generate ML insights.")


        # Leakage Detection
        with tab5:

            leakage = result.get("leakage")

            if leakage and "suspicious_features" in leakage:

                leak_df = pd.DataFrame(leakage["suspicious_features"])

                if not leak_df.empty:

                    fig = px.bar(
                        leak_df,
                        x="feature",
                        y="correlation"
                    )

                    st.plotly_chart(fig, width="stretch")

                    st.warning("Potential data leakage detected.")

                else:
                    st.success("No obvious leakage detected.")

            else:
                st.info("Leakage analysis unavailable.")



# ===================================================
# DATASET COMPARISON
# ===================================================
elif page == "Dataset Comparison":

    st.title("Dataset Comparison")

    col1,col2 = st.columns(2)

    with col1:
        base_file = st.file_uploader("Baseline Dataset", type=["csv"])

    with col2:
        new_file = st.file_uploader("New Dataset", type=["csv"])


    if st.button("Compare") and base_file and new_file:

        df_base = pd.read_csv(base_file)
        df_new = pd.read_csv(new_file)

        drift = detect_data_drift(df_base, df_new)
        version = compare_dataset_versions(df_base, df_new)

        st.subheader("Drift Detection")

        if drift["drift_features"]:

            drift_df = pd.DataFrame(
                drift["drift_features"],
                columns=["Drifted Features"]
            )

            st.dataframe(drift_df, width="stretch")

        else:
            st.success("No statistical drift detected")

        st.subheader("Version Comparison")
        st.dataframe(pd.DataFrame([version]))



# ===================================================
# HISTORY
# ===================================================
elif page == "History":

    st.title("Dataset Monitoring History")

    try:

        resp = requests.get("http://127.0.0.1:8000/api/history")

        if resp.status_code == 200:

            history = resp.json()

            if history:

                hist_df = pd.DataFrame(history)

                st.subheader("Dataset Analysis Records")
                st.dataframe(hist_df, width="stretch")

                if "health_score" in hist_df.columns:

                    st.subheader("Dataset Health Monitoring")

                    fig = px.line(
                        hist_df,
                        x=hist_df.index,
                        y="health_score",
                        markers=True
                    )

                    fig.update_layout(
                        yaxis_range=[0,100]
                    )

                    st.plotly_chart(fig, width="stretch")

            else:
                st.info("No analysis history available")

        else:
            st.error(f"History API returned error: {resp.status_code}")

    except Exception as e:
        st.error(f"Failed to load dataset history: {e}")