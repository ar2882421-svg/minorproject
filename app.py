from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_data.csv"
MODEL_PATH = ROOT / "models" / "student_outcome_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
OUTCOMES = ["Dropout", "Enrolled", "Graduate"]

st.set_page_config(
    page_title="StudentPulse | Academic Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; }
.block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: #101a2d; }
[data-testid="stSidebar"] * { color: #e7edf7 !important; }
.hero { background: linear-gradient(120deg,#14243e 0%,#244c75 55%,#367b91 100%); padding: 28px 32px; border-radius: 22px; color: white; margin-bottom: 20px; }
.hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: 2.15rem; margin: 0 0 7px 0; font-weight: 800; }
.hero p { color: #d8e7f7; margin: 0; font-size: 1rem; }
.eyebrow { text-transform: uppercase; letter-spacing: 2px; color: #8ee3d1; font-size: .72rem; font-weight: 700; margin-bottom: 8px; }
.kpi { background: white; border: 1px solid #e4eaf2; border-radius: 17px; padding: 19px 20px; box-shadow: 0 3px 15px rgba(27,45,75,.035); min-height: 112px; }
.kpi-label { color: #697991; font-size: .82rem; font-weight: 600; margin-bottom: 9px; }
.kpi-value { color: #15243a; font-family: 'Manrope', sans-serif; font-size: 1.85rem; line-height: 1.1; font-weight: 800; }
.kpi-note { color: #8290a4; font-size: .75rem; margin-top: 7px; }
.section-title { color: #192b43; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.18rem; margin: 10px 0 12px 0; }
.small-note { color: #738198; font-size: .82rem; }
div[data-testid="stMetric"] { background: white; padding: 16px; border: 1px solid #e4eaf2; border-radius: 15px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: white; border-radius: 10px 10px 0 0; padding: 10px 18px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    target = next((c for c in df.columns if c.strip().lower() == "target"), None)
    if target:
        df[target] = df[target].astype(str).str.strip()
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def outcome_col(df):
    return next((c for c in df.columns if c.strip().lower() == "target"), None)

def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def card(label, value, note, accent="#2b7188"):
    st.markdown(f"""
    <div class="kpi" style="border-top:3px solid {accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-note">{note}</div>
    </div>""", unsafe_allow_html=True)

def prepare_project():
    try:
        from download_dataset import download_dataset
        from train_model import train_and_save
        with st.spinner("Preparing dataset and model for the first time..."):
            download_dataset()
            st.cache_data.clear()
            st.cache_resource.clear()
            if not MODEL_PATH.exists() or not METRICS_PATH.exists():
                train_and_save()
        st.success("Dataset and model are ready.")
        st.rerun()
    except Exception as e:
        st.error(f"Setup failed: {e}")
        st.info("Check internet access, then run `pip install -r requirements.txt` in the project folder.")

df = load_dataset()
if df is None:
    st.markdown('<div class="hero"><div class="eyebrow">STUDENTPULSE · ACADEMIC INTELLIGENCE</div><h1>Student Performance & Dropout Risk</h1><p>Turn academic data into earlier, more informed student support.</p></div>', unsafe_allow_html=True)
    st.warning("Dataset not found yet. Set up the project once to download the official UCI dataset and train the model.")
    if st.button("🚀 Set up dataset and model", type="primary"):
        prepare_project()
    st.code("python -m pip install -r requirements.txt\nstreamlit run app.py", language="powershell")
    st.stop()

target = outcome_col(df)
if target is None:
    st.error("The dataset does not contain the expected Target column.")
    st.stop()

metrics = load_metrics()
model = load_model()
if model is None or not metrics:
    st.markdown('<div class="hero"><div class="eyebrow">STUDENTPULSE · ACADEMIC INTELLIGENCE</div><h1>Welcome to StudentPulse</h1><p>Your academic risk dashboard is almost ready.</p></div>', unsafe_allow_html=True)
    st.warning("Model artifacts are missing. Run setup to train the model.")
    if st.button("Train model and prepare dashboard", type="primary"):
        prepare_project()
    st.stop()

outcome_counts = df[target].value_counts()
dropout_count = int(outcome_counts.get("Dropout", 0))
dropout_pct = dropout_count / max(len(df), 1) * 100
accuracy = float(metrics.get("test_accuracy", 0)) * 100
features = [c for c in df.columns if c != target]

with st.sidebar:
    st.markdown("## 🎓 StudentPulse")
    st.caption("ACADEMIC RISK INTELLIGENCE")
    st.divider()
    page = st.radio("NAVIGATION", ["Overview", "Student Risk Explorer", "Predict a Student", "Model Insights"], label_visibility="visible")
    st.divider()
    st.markdown("**Data source**")
    st.caption("UCI Machine Learning Repository")
    st.markdown("[Open official dataset ↗](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)")
    st.caption("Records are anonymized. Predictions are decision-support estimates, not definitive judgments.")

st.markdown(f'<div class="hero"><div class="eyebrow">ACADEMIC INTELLIGENCE PLATFORM</div><h1>{page if page != "Overview" else "Student Performance & Early Risk"}</h1><p>Explore outcomes, understand patterns, and identify where student support may be useful.</p></div>', unsafe_allow_html=True)

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: card("TOTAL STUDENT RECORDS", f"{len(df):,}", "Official academic dataset", "#2d8194")
    with c2: card("OBSERVED DROPOUTS", f"{dropout_count:,}", f"{dropout_pct:.1f}% of dataset records", "#e07861")
    with c3: card("MODEL TEST ACCURACY", f"{accuracy:.1f}%", "Held-out test split", "#6b79c9")
    with c4: card("INPUT FEATURES", f"{len(features)}", "Academic and background fields", "#39a58c")
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Student outcome distribution</div>', unsafe_allow_html=True)
        counts = outcome_counts.rename_axis("Outcome").reset_index(name="Students")
        fig = px.bar(counts, x="Outcome", y="Students", color="Outcome",
                     color_discrete_map={"Dropout":"#e07861","Enrolled":"#e5b45d","Graduate":"#3b9d8f"},
                     text="Students")
        fig.update_layout(template="plotly_white", showlegend=False, margin=dict(l=8,r=8,t=15,b=8), height=340,
                          xaxis_title="", yaxis_title="Student records")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Outcome mix</div>', unsafe_allow_html=True)
        fig = px.pie(counts, names="Outcome", values="Students", hole=.66,
                     color="Outcome", color_discrete_map={"Dropout":"#e07861","Enrolled":"#e5b45d","Graduate":"#3b9d8f"})
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(template="plotly_white", showlegend=False, margin=dict(l=10,r=10,t=15,b=10), height=340)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-title">Dataset preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)
    st.caption("This is an observational dataset. Dataset-level dropout share is not the same as an individual student's predicted risk.")

elif page == "Student Risk Explorer":
    st.markdown('<div class="section-title">Filter student outcomes</div>', unsafe_allow_html=True)
    f1, f2 = st.columns([1, 1])
    choices = sorted(df[target].dropna().unique().tolist())
    with f1:
        selected_outcomes = st.multiselect("Outcome", choices, default=choices)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    with f2:
        numeric_feature = st.selectbox("Compare by numeric feature", numeric_cols,
                                       index=numeric_cols.index("Curricular units 1st sem (grade)") if "Curricular units 1st sem (grade)" in numeric_cols else 0)
    filtered = df[df[target].isin(selected_outcomes)].copy()
    if numeric_feature in filtered.columns and not filtered.empty:
        fig = px.box(filtered, x=target, y=numeric_feature, color=target,
                     color_discrete_map={"Dropout":"#e07861","Enrolled":"#e5b45d","Graduate":"#3b9d8f"})
        fig.update_layout(template="plotly_white", height=370, showlegend=False, margin=dict(l=8,r=8,t=20,b=8))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-title">Matching records</div>', unsafe_allow_html=True)
    st.caption(f"{len(filtered):,} records match your selected outcomes.")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download filtered CSV", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_student_records.csv", mime="text/csv")

elif page == "Predict a Student":
    st.markdown("Enter the student's details below. The form uses only real feature columns present in the UCI dataset and the trained model.")
    X = df.drop(columns=[target]).copy()

    def find_feature(exact_name):
        return next((c for c in X.columns if c.strip().lower() == exact_name.lower()), None)

    age_col = find_feature("Age at enrollment")
    scholarship_col = find_feature("Scholarship holder")
    approved_col = find_feature("Curricular units 1st sem (approved)")
    previous_grade_col = find_feature("Previous qualification (grade)")

    # Start with a typical baseline row, calculated from the actual dataset.
    baseline = {}
    for col in X.columns:
        series = X[col].dropna()
        if series.empty:
            baseline[col] = 0
        elif pd.api.types.is_numeric_dtype(X[col]):
            baseline[col] = float(series.median())
            if pd.api.types.is_integer_dtype(X[col]) or np.all(np.isclose(series.astype(float) % 1, 0)):
                baseline[col] = int(round(float(series.median())))
        else:
            baseline[col] = series.mode().iloc[0]

    with st.form("custom_student_prediction_form"):
        st.markdown('<div class="section-title">Student details</div>', unsafe_allow_html=True)
        st.caption("Your entries override the baseline values. Other model features use typical dataset values unless changed in Advanced features.")
        c1, c2 = st.columns(2)
        user_values = {}
        with c1:
            if age_col:
                user_values[age_col] = st.number_input(
                    "Age at enrollment (years)", min_value=15, max_value=100, value=20, step=1,
                    help="Mapped to the dataset's actual 'Age at enrollment' feature."
                )
            if previous_grade_col:
                pmin = float(pd.to_numeric(X[previous_grade_col], errors="coerce").min())
                pmax = float(pd.to_numeric(X[previous_grade_col], errors="coerce").max())
                user_values[previous_grade_col] = st.number_input(
                    "Previous qualification grade", value=12.0, step=1.0,
                    help=f"Actual dataset feature: '{previous_grade_col}'. Observed dataset range: {pmin:g}–{pmax:g}."
                )
            if approved_col:
                amin = int(max(0, pd.to_numeric(X[approved_col], errors="coerce").min()))
                amax = int(pd.to_numeric(X[approved_col], errors="coerce").max())
                user_values[approved_col] = st.number_input(
                    "Curricular units 1st semester (approved)", min_value=amin, max_value=max(amax, 4), value=4, step=1,
                    help="Mapped to the dataset's real first-semester approved-units feature."
                )
        with c2:
            if scholarship_col:
                scholarship_choice = st.selectbox("Scholarship holder?", ["No", "Yes"], index=0)
                # The UCI dataset encodes Scholarship holder as 0 = No, 1 = Yes.
                user_values[scholarship_col] = 1 if scholarship_choice == "Yes" else 0
            else:
                st.info("Scholarship holder is not a feature in the loaded dataset.")
            st.info("Attendance percentage is not a feature in this UCI dataset, so it is not sent to the model. No unsupported attendance field is invented.")

        with st.expander("Advanced features — edit other model inputs"):
            st.caption("The model expects every feature used during training. These remaining fields start at typical values from the dataset; edit any you know.")
            for col in X.columns:
                if col in user_values:
                    continue
                series = X[col].dropna()
                if series.empty:
                    continue
                if pd.api.types.is_numeric_dtype(X[col]):
                    numeric = pd.to_numeric(series, errors="coerce").dropna()
                    if numeric.empty:
                        continue
                    default = float(baseline[col])
                    if col.strip().lower() == "scholarship holder":
                        choice = st.selectbox(f"{col}", ["No", "Yes"], index=1 if int(round(default)) == 1 else 0)
                        user_values[col] = 1 if choice == "Yes" else 0
                    elif numeric.nunique() <= 12 and np.all(np.isclose(numeric % 1, 0)):
                        opts = sorted(int(v) for v in numeric.unique())
                        default_int = int(round(default))
                        if default_int not in opts:
                            opts.append(default_int)
                            opts = sorted(opts)
                        user_values[col] = st.selectbox(f"{col}", opts, index=opts.index(default_int))
                    else:
                        user_values[col] = st.number_input(
                            col, value=float(default),
                            help=f"Observed dataset range: {numeric.min():g}–{numeric.max():g}"
                        )
                else:
                    opts = series.astype(str).value_counts().index.tolist()
                    default = str(baseline[col])
                    if default not in opts:
                        opts.insert(0, default)
                    user_values[col] = st.selectbox(col, opts, index=opts.index(default))

        submitted = st.form_submit_button("🔍 Predict academic outcome", type="primary", use_container_width=True)

    if submitted:
        try:
            # Combine user inputs with dataset-derived baseline values, preserving model column order.
            input_values = dict(baseline)
            input_values.update(user_values)
            input_row = pd.DataFrame([[input_values[col] for col in X.columns]], columns=X.columns)
            # Restore dtypes where possible so the saved sklearn pipeline receives expected columns.
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    input_row[col] = pd.to_numeric(input_row[col], errors="coerce")
            probabilities = model.predict_proba(input_row)[0]
            classes = list(model.classes_)
            ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
            predicted = ranked[0][0]
            p1, p2 = st.columns(2)
            with p1:
                st.metric("Model-predicted outcome", predicted)
            with p2:
                st.metric("Top outcome probability", f"{ranked[0][1]:.1%}")

            if predicted == "Dropout":
                st.warning("Suggested support: offer a mentor check-in, review academic engagement, and discuss available student services.")
            elif predicted == "Enrolled":
                st.info("Suggested support: monitor progress at the next checkpoint and offer help where needed.")
            else:
                st.success("Suggested support: continue routine academic progress monitoring.")

            prob_df = pd.DataFrame({"Outcome": classes, "Estimated probability": probabilities})
            prob_df = prob_df.sort_values("Estimated probability", ascending=True)
            fig = px.bar(prob_df, x="Estimated probability", y="Outcome", orientation="h",
                         text=prob_df["Estimated probability"].map(lambda x: f"{x:.1%}"),
                         color="Outcome", color_discrete_map={"Dropout":"#e07861","Enrolled":"#e5b45d","Graduate":"#3b9d8f"})
            fig.update_layout(template="plotly_white", xaxis_tickformat=".0%", xaxis_title="Model probability estimate", yaxis_title="", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            if previous_grade_col:
                observed_min = pd.to_numeric(X[previous_grade_col], errors="coerce").min()
                observed_max = pd.to_numeric(X[previous_grade_col], errors="coerce").max()
                if not (observed_min <= float(input_values[previous_grade_col]) <= observed_max):
                    st.warning(f"Previous qualification grade {input_values[previous_grade_col]:g} is outside the dataset's observed range ({observed_min:g}–{observed_max:g}). The model still returns an estimate, but treat it cautiously.")
            st.caption("Important: unspecified features use median/mode values from the dataset. These are estimates, not certainty. The dataset includes academic-progress features, so this is not a guaranteed admission-time forecast.")
            with st.expander("Review all values sent to the model"):
                st.dataframe(input_row.T.rename(columns={0: "Model input value"}), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Insights":
    st.markdown('<div class="section-title">Evaluation summary</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Selected model", metrics.get("selected_model", "Not available"))
    b.metric("Test accuracy", f"{accuracy:.1f}%")
    c.metric("Test records", f"{metrics.get('test_rows', 0):,}")
    st.caption("Accuracy is calculated on a held-out test split; precision, recall and F1-score give a more complete view, especially when classes are imbalanced.")
    comparison = pd.DataFrame(metrics.get("model_comparison", []))
    if not comparison.empty:
        st.markdown('<div class="section-title">Model comparison</div>', unsafe_allow_html=True)
        comparison["accuracy_percent"] = comparison["accuracy"] * 100
        fig = px.bar(comparison, x="model", y="accuracy_percent", text=comparison["accuracy_percent"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(marker_color="#3b9d8f")
        fig.update_layout(template="plotly_white", yaxis_title="Test accuracy (%)", xaxis_title="", height=320)
        st.plotly_chart(fig, use_container_width=True)
    report = metrics.get("classification_report", {})
    rows = []
    for label, vals in report.items():
        if isinstance(vals, dict) and all(k in vals for k in ["precision", "recall", "f1-score"]):
            rows.append({"Class": label, "Precision": vals["precision"], "Recall": vals["recall"], "F1-score": vals["f1-score"], "Support": vals.get("support", 0)})
    if rows:
        st.markdown('<div class="section-title">Classification report</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows).style.format({"Precision":"{:.3f}", "Recall":"{:.3f}", "F1-score":"{:.3f}"}), use_container_width=True, hide_index=True)
    cm_path = ROOT / "reports" / "confusion_matrix.png"
    fi_path = ROOT / "reports" / "feature_importance.png"
    c1, c2 = st.columns(2)
    with c1:
        if cm_path.exists():
            st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
            st.image(str(cm_path), use_container_width=True)
    with c2:
        if fi_path.exists():
            st.markdown('<div class="section-title">Top feature importance</div>', unsafe_allow_html=True)
            st.image(str(fi_path), use_container_width=True)
        else:
            st.info("Feature importance chart is available when the selected model is Random Forest.")
    with st.expander("Project methodology and limitations"):
        st.write("The app compares Logistic Regression and Random Forest using a stratified 80/20 train/test split. The better test-accuracy model is saved. The dataset's target is Dropout, Enrolled, or Graduate. Some input features describe academic progress after enrollment; therefore, results are not an admission-time forecast and must not be treated as definitive judgments about individuals.")
        st.write("Dataset source: UCI Machine Learning Repository — Predict Students' Dropout and Academic Success.")
        st.markdown("https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success")

st.divider()
st.markdown('<p class="small-note">STUDENTPULSE · Academic data for supportive decisions · Predictions should be reviewed by faculty and mentors.</p>', unsafe_allow_html=True)
