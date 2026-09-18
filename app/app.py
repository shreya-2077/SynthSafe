import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "customer_data.csv"
SYNTHETIC_FILE = BASE_DIR / "data" / "synthetic" / "synthetic_data.csv"

SUMMARY_FILE = BASE_DIR / "reports" / "synthsafe_summary.csv"
STAT_FILE = BASE_DIR / "reports" / "statistical_comparison.csv"
CORR_FILE = BASE_DIR / "reports" / "correlation_difference.csv"
PRIVACY_FILE = BASE_DIR / "reports" / "privacy_report.csv"


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="SynthSafe",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛡️ SynthSafe")
st.subheader("Synthetic Data Generator + Privacy Analyzer")

st.write(
    "Generate synthetic datasets, evaluate their statistical utility, "
    "and analyze privacy-related proximity indicators."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ SynthSafe Pipeline")

st.sidebar.markdown("""
**Pipeline**

1. 📂 Upload Dataset
2. 🧬 Generate Synthetic Data
3. 📈 Utility Evaluation
4. 🔐 Privacy Analysis
5. 🎯 SynthSafe Summary
6. 📊 Visual Analytics
""")


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.header("📂 Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_FILE, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Dataset uploaded successfully!")

    st.rerun()


# --------------------------------------------------
# LOAD ORIGINAL DATA
# --------------------------------------------------

if RAW_FILE.exists():

    original = pd.read_csv(RAW_FILE)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Records",
        len(original)
    )

    col2.metric(
        "Features",
        len(original.columns)
    )

    col3.metric(
        "Missing Values",
        int(original.isnull().sum().sum())
    )

    with st.expander("👀 Preview Original Dataset"):

        st.dataframe(
            original.head(10),
            use_container_width=True
        )

else:

    st.info("Upload a CSV dataset to begin.")

    st.stop()


# --------------------------------------------------
# RUN SCRIPT FUNCTION
# --------------------------------------------------

def run_script(script_name):

    script_path = BASE_DIR / "src" / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        st.error(f"Error while running {script_name}")

        if result.stderr:
            st.code(result.stderr)

        return False

    return True


# --------------------------------------------------
# PIPELINE BUTTONS
# --------------------------------------------------

st.divider()

st.header("🚀 Run SynthSafe Pipeline")

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🧬 Generate Synthetic Dataset",
        use_container_width=True
    ):

        with st.spinner("Generating synthetic dataset..."):

            if run_script("generator.py"):

                st.success(
                    "Synthetic dataset generated successfully!"
                )

                st.rerun()


with col2:

    if st.button(
        "📈 Run Utility Evaluation",
        use_container_width=True
    ):

        with st.spinner("Evaluating synthetic data..."):

            if run_script("evaluator.py"):

                st.success(
                    "Utility evaluation completed!"
                )

                st.rerun()


col3, col4 = st.columns(2)

with col3:

    if st.button(
        "🔐 Run Privacy Analyzer",
        use_container_width=True
    ):

        with st.spinner("Analyzing privacy indicators..."):

            if run_script("privacy_analyzer.py"):

                st.success(
                    "Privacy analysis completed!"
                )

                st.rerun()


with col4:

    if st.button(
        "🎯 Generate SynthSafe Summary",
        use_container_width=True
    ):

        with st.spinner("Generating final SynthSafe summary..."):

            if run_script("score_engine.py"):

                st.success(
                    "SynthSafe summary generated!"
                )

                st.rerun()


# --------------------------------------------------
# SYNTHETIC DATA
# --------------------------------------------------

if SYNTHETIC_FILE.exists():

    st.divider()

    st.header("🧬 Synthetic Dataset")

    synthetic = pd.read_csv(SYNTHETIC_FILE)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Synthetic Records",
        len(synthetic)
    )

    c2.metric(
        "Synthetic Features",
        len(synthetic.columns)
    )

    c3.metric(
        "Missing Values",
        int(synthetic.isnull().sum().sum())
    )

    with st.expander("👀 Preview Synthetic Dataset"):

        st.dataframe(
            synthetic.head(10),
            use_container_width=True
        )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

if SUMMARY_FILE.exists():

    st.divider()

    st.header("🎯 SynthSafe Summary")

    summary = pd.read_csv(SUMMARY_FILE)

    # Convert summary into dictionary
    if len(summary.columns) >= 2:

        summary_dict = dict(
            zip(
                summary.iloc[:, 0].astype(str),
                summary.iloc[:, 1]
            )
        )

        utility = float(
            summary_dict.get("Utility Score", 0)
        )

        distribution = float(
            summary_dict.get(
                "Distribution Similarity",
                0
            )
        )

        correlation = float(
            summary_dict.get(
                "Correlation Similarity",
                0
            )
        )

        exact_matches = summary_dict.get(
            "Exact Matches",
            0
        )

        duplicates = summary_dict.get(
            "Duplicate Records",
            0
        )

        very_close = summary_dict.get(
            "Very-Close Records",
            0
        )

        very_close_percentage = float(
            summary_dict.get(
                "Very-Close Percentage",
                0
            )
        )


        # ------------------------------------------
        # MAIN SCORE
        # ------------------------------------------

        st.markdown(
            f"""
            <div style="
                padding: 25px;
                border-radius: 15px;
                border: 2px solid #888;
                text-align: center;
                margin-bottom: 25px;
            ">
                <h2>🛡️ SynthSafe Utility Score</h2>
                <h1 style="font-size: 55px;">
                    {utility:.2f} / 100
                </h1>
                <p>
                    Project-defined utility score based on
                    distribution and correlation similarity.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ------------------------------------------
        # METRICS
        # ------------------------------------------

        a, b, c = st.columns(3)

        a.metric(
            "📊 Distribution Similarity",
            f"{distribution:.2f}%"
        )

        b.metric(
            "🔗 Correlation Similarity",
            f"{correlation:.2f}%"
        )

        c.metric(
            "🎯 Exact Matches",
            int(exact_matches)
        )


        d, e, f = st.columns(3)

        d.metric(
            "♻️ Duplicate Records",
            int(duplicates)
        )

        e.metric(
            "🔍 Very-Close Records",
            int(very_close)
        )

        f.metric(
            "🔐 Very-Close %",
            f"{very_close_percentage:.2f}%"
        )


# --------------------------------------------------
# STATISTICAL COMPARISON
# --------------------------------------------------

if STAT_FILE.exists():

    st.divider()

    st.header("📊 Statistical Comparison")

    stats = pd.read_csv(STAT_FILE)

    st.dataframe(
        stats,
        use_container_width=True
    )


# --------------------------------------------------
# CORRELATION DIFFERENCE
# --------------------------------------------------

if CORR_FILE.exists():

    st.divider()

    st.header("🔗 Correlation Difference")

    corr_diff = pd.read_csv(CORR_FILE)

    st.dataframe(
        corr_diff,
        use_container_width=True
    )


# --------------------------------------------------
# VISUAL ANALYTICS
# --------------------------------------------------

if SYNTHETIC_FILE.exists():

    st.divider()

    st.header("📊 Visual Analytics")

    numeric_columns = original.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_columns:

        selected_column = st.selectbox(
            "Select a numerical feature",
            numeric_columns
        )

        fig = plt.figure(
            figsize=(9, 5)
        )

        plt.hist(
            original[selected_column].dropna(),
            bins=25,
            alpha=0.6,
            label="Original"
        )

        plt.hist(
            synthetic[selected_column].dropna(),
            bins=25,
            alpha=0.6,
            label="Synthetic"
        )

        plt.xlabel(selected_column)

        plt.ylabel("Frequency")

        plt.title(
            f"Original vs Synthetic: {selected_column}"
        )

        plt.legend()

        st.pyplot(fig)


# --------------------------------------------------
# PRIVACY REPORT
# --------------------------------------------------

if PRIVACY_FILE.exists():

    st.divider()

    st.header("🔐 Privacy Analysis")

    privacy = pd.read_csv(PRIVACY_FILE)

    st.dataframe(
        privacy,
        use_container_width=True
    )

    st.info(
        "Privacy metrics shown here are proximity-based indicators "
        "and should not be interpreted as formal privacy guarantees."
    )


# --------------------------------------------------
# DOWNLOAD REPORTS
# --------------------------------------------------

st.divider()

st.header("📥 Download Reports")

download_cols = st.columns(4)

if SYNTHETIC_FILE.exists():

    with download_cols[0]:

        st.download_button(
            "Synthetic Dataset",
            data=SYNTHETIC_FILE.read_bytes(),
            file_name="synthetic_data.csv",
            mime="text/csv"
        )


if SUMMARY_FILE.exists():

    with download_cols[1]:

        st.download_button(
            "Summary",
            data=SUMMARY_FILE.read_bytes(),
            file_name="synthsafe_summary.csv",
            mime="text/csv"
        )


if PRIVACY_FILE.exists():

    with download_cols[2]:

        st.download_button(
            "Privacy Report",
            data=PRIVACY_FILE.read_bytes(),
            file_name="privacy_report.csv",
            mime="text/csv"
        )


if STAT_FILE.exists():

    with download_cols[3]:

        st.download_button(
            "Statistics",
            data=STAT_FILE.read_bytes(),
            file_name="statistical_comparison.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SynthSafe — Synthetic Data Generation, Utility Evaluation "
    "and Privacy Analysis"
)