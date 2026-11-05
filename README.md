# StudentPulse — Student Performance & Early Dropout Risk Dashboard

A Streamlit dashboard using the UCI Student Dropout and Academic Success dataset. It includes a KPI overview, interactive charts, outcome filters, a student prediction demo, and model evaluation insights.

## Run in VS Code on Windows

1. Extract the ZIP.
2. Open the **inner** `StudentRiskDashboard` folder in VS Code (the folder containing `app.py` and `requirements.txt`).
3. In the VS Code terminal, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Launch the dashboard:

```powershell
python -m streamlit run app.py
```

Streamlit will show a local URL, normally `http://localhost:8501`, and may open your browser automatically. If the dataset is missing, click **Set up dataset and model** on the dashboard. First-time setup needs internet access.

## Dashboard pages

- **Overview:** KPIs, outcome distribution, outcome mix and dataset preview
- **Student Risk Explorer:** Filter outcomes, compare numeric features and export filtered records
- **Predict a Student:** Enter custom values for actual dataset features (age, previous qualification grade, first-semester approved units, scholarship status); edit remaining model inputs in Advanced features; view predicted class/probabilities. Unsupported attendance percentage is not invented.
- **Model Insights:** Model comparison, held-out test metrics, classification report and saved charts

## Dataset

UCI Machine Learning Repository — Predict Students' Dropout and Academic Success (dataset 697):
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

The target classes are `Dropout`, `Enrolled`, and `Graduate`. Some predictors describe academic progress after enrollment, so this is not a guaranteed admission-time dropout forecast. Treat model predictions as decision-support estimates, not labels for judging students.
