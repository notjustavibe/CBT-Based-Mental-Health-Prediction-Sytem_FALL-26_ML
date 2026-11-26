# MindCheck

An interactive machine learning pipeline that screens self-reported stress, anxiety and mood into four risk levels: **Normal**, **Mild**, **Moderate** and **Severe**. It trains and compares three models, then generates a Cognitive Behavioral Report with domain scores, practical advice and country-specific helplines (including Pakistan).

> **Important:** MindCheck is an educational screening project. It is not a diagnostic tool and does not replace a qualified mental health professional. If you or someone you know is in crisis, contact local emergency services or a helpline immediately.

## Features

- Merges two public survey datasets into one 19-feature representation
- Feature engineering with 15 base features and 4 interaction features
- Compares Linear Regression (rounded), Logistic Regression and Random Forest on the same stratified 80/20 split
- Evaluation with accuracy, confusion matrices and a per-class classification report
- Saves trained models as reusable `.pkl` files
- Interactive sliders and drop-downs (ipywidgets) that produce an HTML report card and charts
- Four-domain breakdown: Emotional Health, Social Wellbeing, Stress Resilience, Future Outlook
- SQLite storage of user profiles and assessment history
- Self-healing setup cell that installs any missing Python library

## Repository Contents

| File | Description |
|---|---|
| `MindCheck_Interactive_Pipeline.ipynb` | Complete 8-step pipeline |
| `logistic_regression_model.pkl` | Saved baseline Logistic Regression package |
| `random_forest_model.pkl` | Saved Random Forest package |
| `db.py` | SQLite helper module imported by the notebook (add to the repo root) |
| `survey.csv` | Mental Health in Tech Survey data (download separately) |
| `Mental_Health_Dataset.csv` | Global Mental Health Dataset (download separately) |

Running the notebook also generates `model.pkl` (a copy of the Random Forest package) and `mindcheck.db` (the SQLite database).

## Pipeline Overview

1. Auto-install and import required libraries
2. Load both datasets (the larger one is sampled to 30,000 rows, `random_state=42`)
3. Translate answers to numeric features and build the risk label
4. Train baseline models (Linear and Logistic Regression)
5. Train the Random Forest (100 trees, max depth 20) and compare results
6. Prediction engine and report generation from 1 to 5 slider answers
7. SQLite database for users and saved reports
8. Interactive control panel and history viewer

## Getting Started

### Requirements

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- numpy, pandas, matplotlib, seaborn, scikit-learn, joblib, ipywidgets (the first notebook cell installs any that are missing)

### Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook MindCheck_Interactive_Pipeline.ipynb
```

Place `survey.csv`, `Mental_Health_Dataset.csv` and `db.py` in the same folder as the notebook, then run all cells from top to bottom. The control panel in Step 7 appears below its cell and generates a report when you press the button.

### Loading a saved model

```python
import joblib

package = joblib.load("random_forest_model.pkl")
model = package["model"]
columns = package["columns"]   # the 19 feature names, in order
```

## Results

Reported on the held-out test set (8,015 rows) in the notebook:

| Model | Accuracy |
|---|---|
| Linear Regression (rounded) | 81.66% |
| Logistic Regression | 84.17% |
| Random Forest | 97.16% |

## Limitations

Please read these before interpreting the accuracy figures.

- **Labels are rule-derived, not clinical.** Risk levels come from hand-written scoring formulas applied to the same answers used as model inputs, so the models largely learn to reproduce those formulas. Accuracy measures agreement with the rules, not clinical detection.
- **Possible data leakage.** The smaller dataset is repeated eight times before the train and test split, so duplicate rows can appear in both sets and inflate test accuracy. A fairer approach is to split first and oversample only the training data.
- **Training and live inputs differ.** The live questionnaire maps 1 to 5 sliders to features using fixed weights that were not calibrated against the training data, and some training features use default values.
- **Dataset bias.** Neither dataset is specific to Pakistan, and the tech-worker survey covers one profession and region.
- **Helplines can change.** Verify the numbers listed in the notebook before relying on them.

## Possible Improvements

- Split before oversampling and evaluate with cross-validation
- Use validated instruments (PHQ-9, GAD-7) for labels and questions
- Probability calibration and feature importance or SHAP explanations
- A web or mobile front end loading `model.pkl`

## Tech Stack

Python, Jupyter, NumPy, pandas, scikit-learn, joblib, matplotlib, seaborn, ipywidgets, SQLite

## Data Sources

- Open Sourcing Mental Illness (OSMI), Mental Health in Tech Survey
- Global Mental Health Dataset (public survey data on Kaggle; confirm the exact source and licence before publishing)

## Author

Muhammad Qasim Ashfaq, BS Artificial Intelligence, Superior University

## License

Add a license of your choice (for example MIT) and confirm that the dataset licences permit redistribution before uploading the CSV files.
