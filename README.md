# Classical Test / ITR Streamlit Dashboard

This small Streamlit app lets you upload a CSV of item responses (rows = respondents, columns = items) and computes:

- Respondent total scores (classical test score)
- Corrected item-total correlations (ITR)
- Cronbach's alpha

Files added:

- [app.py](app.py): Streamlit app
- [sample_responses.csv](sample_responses.csv): sample data you can use to test

Run locally:

```powershell
cd path\to\project
streamlit run app.py
```

Usage:

- Upload your CSV using the sidebar or press "Load sample_responses.csv" to load the sample file.
- Choose between "Respondent scores (Classical)" or "Item stats (ITR)" from the sidebar.
- Download results using the provided download buttons.

If you want additional metrics (item difficulty, ICC plots, or advanced reporting), tell me what you need and I can add it.
