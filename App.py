import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0b1120;
    --bg-soft: #0f172a;
    --card: #131c31;
    --card-hover: #17223b;
    --border: rgba(148, 163, 184, 0.14);
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-2: #6366f1;
    --good: #22c55e;
    --bad: #ef4444;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background:
        radial-gradient(900px 500px at 85% -10%, rgba(99,102,241,0.18), transparent 60%),
        radial-gradient(800px 500px at -10% 0%, rgba(56,189,248,0.12), transparent 55%),
        var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Typography */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff 0%, #7dd3fc 60%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 0.1rem;
}
h2, h3 {
    color: #ffffff;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
h2 {
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
[data-testid="stCaptionContainer"], .stCaption {
    color: var(--muted) !important;
    font-size: 0.95rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--card) 0%, var(--bg-soft) 100%);
    border: 1px solid var(--border);
    padding: 20px 22px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    transition: transform .2s ease, border-color .2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(56, 189, 248, 0.45);
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #ffffff;
    font-weight: 700;
    font-size: 2rem !important;
}

/* Charts / dataframes as cards */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0e1628 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text);
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 6px;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 12px;
    transition: all .15s ease;
    width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(56, 189, 248, 0.08);
    border-color: var(--border);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(56,189,248,0.18), rgba(99,102,241,0.14));
    border-color: rgba(56, 189, 248, 0.4);
}
[data-testid="stSidebar"] hr {
    border-color: var(--border);
}

/* Alerts (info / error / success) */
[data-testid="stAlert"] {
    background: rgba(56, 189, 248, 0.07);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    color: var(--text);
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background: var(--card);
    border: 1.5px dashed rgba(56, 189, 248, 0.4);
    border-radius: 16px;
    padding: 1.4rem;
    transition: all .2s ease;
}
[data-testid="stFileUploader"] section:hover {
    background: var(--card-hover);
    border-color: var(--accent);
}

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(56, 189, 248, 0.25);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFileUploader"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(56, 189, 248, 0.4);
    color: #ffffff;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #243049; border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)


# Shared chart styling (visual only)
def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        title_font=dict(size=18, color="#ffffff"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.10)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.10)")
    return fig


TRAFFIC_COLORS = {
    "Normal": "#22c55e",
    "Attack": "#ef4444",
    "BENIGN": "#22c55e",
    "ATTACK": "#ef4444"
}

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/intrusion_model.pkl")

st.title("🛡️ Network Intrusion Detection System")
st.caption("Machine Learning Based Network Traffic Security Analysis")

# Check model availability
if not os.path.exists("models/intrusion_model.pkl"):
    st.error("Model not found. Run train_model.py first.")
    st.stop()

saved = load_model()

model = saved["model"]
features = saved["features"]

# Sidebar navigation
st.sidebar.markdown("## 🛡️ IDS Console")
st.sidebar.caption("Network security analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Traffic Analysis",
        "Threat Detection",
        "Model Information"
    ]
)

# Load sample data for overview
df = pd.read_csv("data/network_traffic_sample.csv")

df["Traffic Type"] = df["Label"].apply(
    lambda x: "Normal" if x == "BENIGN" else "Attack"
)

if page == "Dashboard Overview":

    st.header("Security Overview")

    total = len(df)
    normal = (df["Traffic Type"] == "Normal").sum()
    attacks = (df["Traffic Type"] == "Attack").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Traffic Records", total)
    col2.metric("Normal Records", normal)
    col3.metric("Attack Records", attacks)

    st.subheader("Traffic Distribution")

    counts = df["Traffic Type"].value_counts().reset_index()
    counts.columns = ["Traffic Type", "Count"]

    fig = px.pie(
        counts,
        names="Traffic Type",
        values="Count",
        hole=0.55,
        title="Normal vs Attack Traffic",
        color="Traffic Type",
        color_discrete_map=TRAFFIC_COLORS
    )
    fig.update_traces(
        textinfo="percent+label",
        marker=dict(line=dict(color="#0b1120", width=3))
    )
    style_fig(fig)

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Overview statistics are based on the included "
        "development sample dataset."
    )

elif page == "Traffic Analysis":

    st.header("Network Traffic Analysis")

    st.subheader("Attack Category Distribution")

    fig = px.histogram(
        df,
        x="Label",
        color="Label",
        title="Network Attack Categories",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    style_fig(fig)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flow Duration Analysis")

    fig2 = px.histogram(
        df,
        x="Flow Duration",
        color="Traffic Type",
        title="Network Flow Duration",
        color_discrete_map=TRAFFIC_COLORS
    )
    style_fig(fig2)

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Traffic Records")
    st.dataframe(df.head(50), use_container_width=True)

elif page == "Threat Detection":

    st.header("Upload Network Traffic Data")

    uploaded_file = st.file_uploader(
        "Upload a compatible network traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            missing = [
                col for col in features
                if col not in uploaded_df.columns
            ]

            if missing:
                st.error(f"Missing columns: {missing}")

            else:

                X = uploaded_df[features].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                predictions = model.predict(X)

                results = uploaded_df.copy()

                results["Prediction"] = [
                    "ATTACK" if value == 1 else "BENIGN"
                    for value in predictions
                ]

                total_records = len(results)
                attack_count = (
                    results["Prediction"] == "ATTACK"
                ).sum()

                normal_count = total_records - attack_count

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Records", total_records)
                col2.metric("Detected Attacks", attack_count)
                col3.metric("Normal Traffic", normal_count)

                st.subheader("Detection Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                chart_data = results[
                    "Prediction"
                ].value_counts().reset_index()

                chart_data.columns = ["Prediction", "Count"]

                fig = px.pie(
                    chart_data,
                    names="Prediction",
                    values="Count",
                    hole=0.55,
                    title="Predicted Traffic Distribution",
                    color="Prediction",
                    color_discrete_map=TRAFFIC_COLORS
                )
                fig.update_traces(
                    textinfo="percent+label",
                    marker=dict(line=dict(color="#0b1120", width=3))
                )
                style_fig(fig)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                csv_data = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Detection Report",
                    data=csv_data,
                    file_name="intrusion_detection_report.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Unable to process file: {e}")

elif page == "Model Information":

    st.header("Machine Learning Model Information")

    st.write("Selected Model:", saved["model_name"])

    st.write("Number of Features:", len(features))

    st.subheader("Features Used")

    st.write(features)

    st.info(
        "The system performs offline classification of uploaded "
        "network traffic records. It does not monitor live traffic."
    )

st.sidebar.markdown("---")
st.sidebar.caption("ML Network Intrusion Detector | Project")