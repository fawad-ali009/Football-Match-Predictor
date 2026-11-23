# ⚽ Football Match Predictor

A Machine Learning project that predicts football match outcomes (**Home Win, Draw, or Away Win**) using historical football statistics from Europe's top five leagues.

The system analyzes previous team performances, recent form, attacking and defensive statistics, and home/away records to generate match predictions.

## Project Overview

The goal of this project is to develop a football outcome prediction system using historical match data and Machine Learning techniques.

Users can select two teams (Home and Away), and the system automatically generates the required historical features and predicts the most likely match outcome.

## Dataset

The primary dataset was obtained from the Kaggle **Football Database** and contains historical information from Europe's top five leagues:

- English Premier League
- Spanish La Liga
- Italian Serie A
- German Bundesliga
- French Ligue 1

**Dataset information:**

| Description | Value |
|---|---|
| Usable matches | 12,679 |
| Teams | 146 |
| Model input features | 32 |
| Prediction classes | Home Win (W), Draw (D), Away Win (L) |

Three main datasets are used: `games.csv`, `stats.csv`, and `teams.csv`.

Additional real match results from the 2021/22 season were collected from [Football-Data](https://www.football-data.co.uk/) for external evaluation.

## Technologies Used

- **Python** — Main programming language
- **Pandas and NumPy** — Data manipulation and numerical operations
- **Matplotlib and Seaborn** — Data visualization
- **Scikit-learn** — Machine Learning and model evaluation
- **Joblib** — Saving and loading trained models
- **Git and GitHub** — Version control

## Machine Learning Pipeline

The project follows a modular Machine Learning workflow:

1. **Data Collection:** Load football match results, team information, and match statistics.
2. **Data Cleaning:** Handle missing values, remove duplicates, and prepare datasets.
3. **Exploratory Data Analysis:** Examine distributions, statistics, and relationships between features.
4. **Feature Engineering:** Generate 32 historical features, including recent form, goals, xGoals, shots, and home/away win rates.
5. **Data Preprocessing:** Perform chronological data splitting, missing-value imputation, and feature scaling.
6. **Model Training:** Train and compare Machine Learning classifiers.
7. **Hyperparameter Optimization:** Optimize models using GridSearchCV and time-series cross-validation.
8. **Model Evaluation:** Evaluate predictions using accuracy, precision, recall, F1-score, and confusion matrices.
9. **Feature Importance:** Analyze the contribution of individual historical features.
10. **Prediction:** Generate match predictions using the saved trained model.

## Machine Learning Models

Three classification algorithms were trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- Histogram Gradient Boosting Classifier

**Logistic Regression** was selected as the final model based on validation macro F1-score.

## Model Performance

### Final Chronological Test

The final model was evaluated on 2,327 reserved historical matches.

| Metric | Result |
|---|---|
| Accuracy | **50.75%** |
| Macro F1-score | **0.4803** |
| Testing matches | 2,327 |

### External Evaluation — 2021/22 Season

An additional evaluation was conducted using 100 real matches selected from the five major European leagues, with 20 matches from each league.

| Metric | Result |
|---|---|
| Total matches | 100 |
| Correct predictions | 52 |
| Accuracy | **52.00%** |
| Macro F1-score | **0.4909** |

**Accuracy by league:**

| League | Accuracy |
|---|---|
| Premier League | 45% |
| La Liga | 65% |
| Serie A | 45% |
| Bundesliga | 50% |
| Ligue 1 | 55% |

The external evaluation uses a frozen historical team-statistics snapshot rather than updating features after every 2021/22 match.

## Match Prediction

The standalone prediction system allows users to enter two teams without manually providing 32 input features.

**Example inputs:**

- Home Team: Real Madrid
- Away Team: Barcelona

The system automatically:

1. Identifies the selected teams.
2. Retrieves their historical match statistics.
3. Generates 32 input features.
4. Loads the saved Logistic Regression model.
5. Predicts Home Win, Draw, or Away Win.
6. Displays model probability estimates for all three outcomes.

The system predicts the **match outcome, not the exact final score**.

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/fawad-ali009/Football-Match-Predictor.git
cd Football-Match-Predictor
```

**2. Install the required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

**3. Run the standalone predictor**

```bash
python predict_match.py
```

The program will ask for the home team and away team, then display the prediction.

**Note:** The required processed datasets and saved model must be present in their expected folders. The saved model was created with scikit-learn and should be loaded in a compatible Python environment.

## Project Structure

```text
Football-Match-Predictor/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_optimization.py
│   ├── feature_importance.py
│   ├── feature_generator.py
│   ├── season_prediction.py
│   └── season_evaluation.py
│
├── models/
│   └── final_football_predictor.joblib
│
├── main.py
├── predict_match.py
└── README.md
```

## Limitations

- Draws remain more difficult to predict than wins or losses.
- The current historical statistics are outdated for live 2026 fixtures.
- The model's probability estimates have not been calibrated.
- Predictions are based on historical team performance and do not incorporate current injuries, transfers, or live team news.
- The 2021/22 external evaluation covers a sample of 100 fixtures rather than the complete season.

## Future Improvements

- Develop a web interface using HTML, CSS, and JavaScript.
- Integrate the Machine Learning model with the website using Flask.
- Incorporate updated football statistics for current-season predictions.
- Improve prediction accuracy with additional pre-match features.

## Project Status

**Core Machine Learning pipeline: Completed**

**Web interface and integration: Planned**

## Author

**Fawad Ali**

Machine Learning Semester Project — Sukkur IBA University