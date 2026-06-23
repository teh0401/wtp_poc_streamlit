# Clarifier pH Prediction POC

Streamlit proof of concept for predicting `pH_9021_Clarifier` 30 minutes ahead and converting the prediction into a chemical adjustment level.

## Local Setup

```powershell
cd poc_streamlit_webUI
python -m pip install -r requirements.txt
python scripts/export_model.py
streamlit run app.py
```

## Deploy Notes

For Streamlit Community Cloud, publish this folder to GitHub with the generated `artifacts/` files included:

- `artifacts/model.joblib`
- `artifacts/metadata.json`
- `artifacts/feature_list.json`
- `artifacts/adjustment_rules.json`
- `artifacts/sample_inputs.csv`

The app is a POC only. Production sensor polling, authentication, alert delivery, and database storage are intentionally out of scope.
