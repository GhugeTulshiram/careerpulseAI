from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score


# ============================================================
# CAREER PULSE AI - PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "clean_jobs.csv"
MODELS = ROOT / "models"

# Create models folder if it does not exist
MODELS.mkdir(exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("\n======================================")
print("      CAREER PULSE AI - ML TRAINING")
print("======================================\n")


# ============================================================
# SALARY PREDICTION MODEL
# ============================================================

def train_salary(df):

    print("Training Salary Prediction Model...")

    # Features required for salary prediction
    features = [
        "job_title",
        "location",
        "experience",
        "industry"
    ]

    # Remove rows having missing values
    work = df.dropna(
        subset=features + ["salary_avg"]
    ).copy()

    # Check dataset size
    if len(work) < 5:

        print(
            "Not enough data for salary prediction model."
        )

        return None

    X = work[features]
    y = work["salary_avg"]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    prep = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                features
            )
        ]
    )

    # --------------------------------------------------------
    # Random Forest Regression Pipeline
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "prep",
                prep
            ),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        pred
    )

    print(
        f"Salary MAE: {mae:,.0f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    salary_model_path = (
        MODELS / "salary_model.pkl"
    )

    joblib.dump(
        model,
        salary_model_path
    )

    print(
        "Salary model saved successfully."
    )

    return mae


# ============================================================
# JOB CLASSIFICATION MODEL
# ============================================================

def train_job_classifier(df):

    print("\nTraining Job Classification Model...\n")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "description",
        "industry"
    ]

    # Check columns
    for column in required_columns:

        if column not in df.columns:

            print(
                f"Missing column: {column}"
            )

            print(
                "Job classification model skipped."
            )

            return None

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    work = df.dropna(
        subset=required_columns
    ).copy()

    # Remove empty descriptions
    work = work[
        work["description"]
        .astype(str)
        .str.strip()
        .str.len()
        > 0
    ]

    # --------------------------------------------------------
    # Display original industry counts
    # --------------------------------------------------------

    print("Industry counts:")

    print(
        work["industry"].value_counts()
    )

    # --------------------------------------------------------
    # Remove industries having less than 2 records
    #
    # Why?
    # Logistic Regression needs enough samples
    # for training/testing.
    # --------------------------------------------------------

    industry_counts = (
        work["industry"]
        .value_counts()
    )

    valid_industries = (
        industry_counts[
            industry_counts >= 2
        ]
        .index
    )

    work = work[
        work["industry"].isin(
            valid_industries
        )
    ].copy()

    print("\nIndustries used for training:")

    print(
        work["industry"].value_counts()
    )

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    num_classes = (
        work["industry"]
        .nunique()
    )

    # --------------------------------------------------------
    # Check if classification is possible
    # --------------------------------------------------------

    if num_classes < 2:

        print(
            "\nNot enough industry categories."
        )

        print(
            "Job classification model skipped."
        )

        return None

    if len(work) < num_classes * 2:

        print(
            "\nNot enough records for "
            "reliable classification."
        )

        print(
            "Job classification model skipped."
        )

        return None

    # --------------------------------------------------------
    # Calculate test size
    #
    # Test set must have at least one
    # record for every class.
    # --------------------------------------------------------

    test_size = max(
        num_classes,
        int(len(work) * 0.25)
    )

    # Make sure test size is smaller
    # than complete dataset

    if test_size >= len(work):

        test_size = len(work) - 1

    # --------------------------------------------------------
    # Display split information
    # --------------------------------------------------------

    print(
        f"\nUsing {len(work)} records "
        "for classification"
    )

    print(
        f"Number of classes: {num_classes}"
    )

    print(
        f"Test records: {test_size}"
    )

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        work["description"],

        work["industry"],

        test_size=test_size,

        random_state=42,

        stratify=work["industry"]
    )

    # --------------------------------------------------------
    # TF-IDF + Logistic Regression
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2)
                )
            ),

            (
                "clf",
                LogisticRegression(
                    max_iter=2000
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        pred
    )

    print(
        f"\nJob classification accuracy: "
        f"{accuracy:.2%}"
    )

    # --------------------------------------------------------
    # Save classifier
    # --------------------------------------------------------

    classifier_path = (
        MODELS / "job_classifier.pkl"
    )

    joblib.dump(
        model,
        classifier_path
    )

    print(
        "Job classifier saved successfully."
    )

    return accuracy


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA.exists():

        print(
            "ERROR: clean_jobs.csv not found!"
        )

        print(
            f"Expected location:\n{DATA}"
        )

        raise SystemExit

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA
    )

    print(
        f"Dataset loaded: {len(df)} records"
    )

    # --------------------------------------------------------
    # Train Salary Model
    # --------------------------------------------------------

    train_salary(
        df
    )

    # --------------------------------------------------------
    # Train Job Classifier
    # --------------------------------------------------------

    train_job_classifier(
        df
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n======================================")
    print("          ML TRAINING COMPLETE")
    print("======================================")

    print("\nModels folder:")

    print(
        MODELS
    )

    print("\nGenerated files:")

    salary_model = (
        MODELS / "salary_model.pkl"
    )

    classifier_model = (
        MODELS / "job_classifier.pkl"
    )

    if salary_model.exists():

        print(
            "✓ salary_model.pkl"
        )

    if classifier_model.exists():

        print(
            "✓ job_classifier.pkl"
        )

    print()