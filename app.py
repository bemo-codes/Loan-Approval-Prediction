###############################################################################
#
#   Loan Approval Prediction - Ensemble Machine Learning Dashboard
#
###############################################################################

# =============================================================================
#   Importing required libraries
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


# =============================================================================
#   Application configuration
# =============================================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)


# =============================================================================
#   Dataset configuration
# =============================================================================

DATA_FILE = "loan_approval_dataset.csv"

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CATEGORICAL_FEATURES = [
    "education",
    "self_employed",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "loan_status"


# =============================================================================
#   Function name: normalize_columns
#   Input: df (DataFrame)
#   Description: Standardizes column names and text fields.
#   Output: Cleaned DataFrame
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().str.lower()

    return df


# =============================================================================
#   Function name: load_data
#   Input: data_path (string)
#   Description: Loads and validates the loan approval dataset.
#   Output: Cleaned DataFrame
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(data_path):
    df = pd.read_csv(data_path)
    df = normalize_columns(df)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    # Remove ID because it identifies a row but does not describe credit risk.
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Convert target labels to 1 = Approved and 0 = Rejected.
    target_values = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
    target_map = {
        "approved": 1,
        "rejected": 0,
        "1": 1,
        "0": 0,
    }
    df[TARGET_COLUMN] = target_values.map(target_map)

    # Convert numerical columns and mark invalid values as missing.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Remove rows where every feature is missing.
    df = df.dropna(subset=FEATURE_COLUMNS, how="all")

    return df


# =============================================================================
#   Function name: create_preprocessor
#   Input: None
#   Description: Builds preprocessing for numeric and categorical features.
#   Output: ColumnTransformer
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def create_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# =============================================================================
#   Function name: build_models
#   Input: None
#   Description: Creates individual models and the hard Voting Classifier.
#   Output: Dictionary of model pipelines
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def build_models():
    decision_tree = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    knn = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance",
                ),
            ),
        ]
    )

    voting = VotingClassifier(
        estimators=[
            ("logistic", logistic_regression),
            ("decision_tree", decision_tree),
            ("knn", knn),
        ],
        voting="hard",
    )

    return {
        "Decision Tree": decision_tree,
        "Logistic Regression": logistic_regression,
        "K-Nearest Neighbors": knn,
        "Voting Classifier": voting,
    }


# =============================================================================
#   Function name: train_models
#   Input: X_train (DataFrame), y_train (Series)
#   Description: Trains all individual models and the Voting Classifier.
#   Output: Dictionary containing fitted models
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train):
    models = build_models()

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# =============================================================================
#   Function name: evaluate_models
#   Input: models, X_test, y_test
#   Description: Calculates evaluation metrics for each trained model.
#   Output: Results DataFrame and predictions dictionary
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def evaluate_models(models, X_test, y_test):
    results = []
    predictions = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred

        results.append(
            {
                "Model": model_name,
                "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
                "Precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, y_pred, zero_division=0
                ),
            }
        )

    return pd.DataFrame(results), predictions


# =============================================================================
#   Function name: main
#   Input: None
#   Description: Builds the Streamlit dashboard and loan prediction workflow.
#   Output: Streamlit application
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def main():

    st.title("🏦 Loan Approval Prediction")
    st.caption(
        "Ensemble Machine Learning Dashboard | Applicant Risk Analysis"
    )
    st.write(
        "Explore loan applicant profiles, compare classification models, "
        "and estimate whether an application resembles historically "
        "approved or rejected cases in the supplied dataset."
    )

    # -------------------------------------------------------------------------
    #   Load dataset
    # -------------------------------------------------------------------------

    data_path = Path(__file__).resolve().parent / DATA_FILE

    if not data_path.exists():
        st.error(
            f"`{DATA_FILE}` was not found. Place the CSV beside `app.py`."
        )
        st.stop()

    try:
        df = load_data(str(data_path))
    except Exception as error:
        st.error(f"Unable to load dataset: {error}")
        st.stop()

    if df.empty or df[TARGET_COLUMN].nunique() < 2:
        st.error("The dataset must contain both approved and rejected records.")
        st.stop()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # -------------------------------------------------------------------------
    #   Sidebar
    # -------------------------------------------------------------------------

    st.sidebar.header("⚙️ Model Configuration")
    test_size = st.sidebar.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    # -------------------------------------------------------------------------
    #   Train-test split and training
    # -------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42,
        stratify=y,
    )

    with st.spinner("Training classification models..."):
        models = train_models(X_train, y_train)
        results, predictions = evaluate_models(
            models, X_test, y_test
        )

    # -------------------------------------------------------------------------
    #   Key metrics
    # -------------------------------------------------------------------------

    total_applications = len(df)
    approved_count = int((y == 1).sum())
    rejected_count = int((y == 0).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", f"{total_applications:,}")
    col2.metric("Approved", f"{approved_count:,}")
    col3.metric("Rejected", f"{rejected_count:,}")
    col4.metric(
        "Approval Rate",
        f"{approved_count / total_applications * 100:.1f}%",
    )

    # -------------------------------------------------------------------------
    #   Dashboard tabs
    # -------------------------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📁 Dataset Overview",
            "📊 Applicant Insights",
            "🤖 Model Comparison",
            "🔮 Loan Predictor",
        ]
    )

    # =========================================================================
    #   Tab 1: Dataset Overview
    # =========================================================================

    with tab1:
        st.subheader("Loan Application Dataset")

        left, right = st.columns([2, 1])

        with left:
            st.write("**Dataset Preview**")
            st.dataframe(df.head(15), use_container_width=True)

        with right:
            st.write("**Loan Approval Distribution**")
            counts = y.value_counts().reindex([0, 1], fill_value=0)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(
                ["Rejected", "Approved"],
                counts.values,
            )
            ax.set_ylabel("Number of applications")
            ax.set_title("Approval vs Rejection")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.write("**Descriptive Statistics**")
        st.dataframe(
            df[NUMERIC_FEATURES].describe().T,
            use_container_width=True,
        )

        st.write("**Missing Values**")
        missing = df[FEATURE_COLUMNS].isna().sum().rename("Missing Values")
        st.dataframe(missing.to_frame(), use_container_width=True)

    # =========================================================================
    #   Tab 2: Applicant Insights
    # =========================================================================

    with tab2:
        st.subheader("Applicant Financial Insights")

        st.write("**Correlation Between Numerical Features**")
        correlation_data = df[NUMERIC_FEATURES + [TARGET_COLUMN]].copy()
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(
            correlation_data.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        left, right = st.columns(2)

        with left:
            st.write("**Approval Rate by Education**")
            education_rate = (
                df.groupby("education")[TARGET_COLUMN]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )
            st.bar_chart(education_rate)

        with right:
            st.write("**Approval Rate by Employment Type**")
            employment_rate = (
                df.groupby("self_employed")[TARGET_COLUMN]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )
            st.bar_chart(employment_rate)

        st.write("**CIBIL Score Distribution by Loan Status**")
        plot_df = df.copy()
        plot_df["Loan Status"] = plot_df[TARGET_COLUMN].map(
            {0: "Rejected", 1: "Approved"}
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(
            data=plot_df,
            x="cibil_score",
            hue="Loan Status",
            bins=25,
            kde=True,
            ax=ax,
        )
        ax.set_title("CIBIL Score vs Loan Approval Status")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # =========================================================================
    #   Tab 3: Model Comparison
    # =========================================================================

    with tab3:
        st.subheader("Classification Model Performance")

        st.dataframe(
            results.style.format(
                {
                    "Accuracy (%)": "{:.2f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1 Score": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "Metrics are calculated on the held-out test set. "
            "Compare precision, recall and F1-score in addition to accuracy."
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(
            data=results,
            x="Model",
            y="Accuracy (%)",
            ax=ax,
        )
        ax.set_ylim(0, 100)
        ax.set_title("Accuracy Comparison")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        selected_model = st.selectbox(
            "Select a model to inspect",
            list(models.keys()),
        )

        matrix = confusion_matrix(
            y_test,
            predictions[selected_model],
            labels=[0, 1],
        )

        left, right = st.columns(2)

        with left:
            st.write("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.write("**Classification Report**")
            report = classification_report(
                y_test,
                predictions[selected_model],
                labels=[0, 1],
                target_names=["Rejected", "Approved"],
                zero_division=0,
            )
            st.code(report)

    # =========================================================================
    #   Tab 4: Loan Approval Predictor
    # =========================================================================

    with tab4:
        st.subheader("Check a Loan Application")
        st.write(
            "Enter an applicant profile to see how the selected model "
            "classifies it based on patterns learned from this dataset."
        )

        selected_model = st.selectbox(
            "Choose prediction model",
            list(models.keys()),
            key="prediction_model",
        )

        with st.form("loan_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                dependents = st.number_input(
                    "Number of Dependents",
                    min_value=0,
                    max_value=10,
                    value=int(df["no_of_dependents"].median()),
                    step=1,
                )
                education = st.selectbox(
                    "Education",
                    options=sorted(df["education"].dropna().unique()),
                    format_func=lambda value: value.title(),
                )
                self_employed = st.selectbox(
                    "Self Employed",
                    options=sorted(df["self_employed"].dropna().unique()),
                    format_func=lambda value: value.title(),
                )
                income = st.number_input(
                    "Annual Income (₹)",
                    min_value=0,
                    value=int(df["income_annum"].median()),
                    step=100000,
                )
                loan_amount = st.number_input(
                    "Requested Loan Amount (₹)",
                    min_value=0,
                    value=int(df["loan_amount"].median()),
                    step=100000,
                )
                loan_term = st.number_input(
                    "Loan Term (Years)",
                    min_value=1,
                    max_value=50,
                    value=int(df["loan_term"].median()),
                    step=1,
                )

            with col2:
                cibil_score = st.number_input(
                    "CIBIL Score",
                    min_value=0,
                    max_value=1000,
                    value=int(df["cibil_score"].median()),
                    step=1,
                )
                residential_assets = st.number_input(
                    "Residential Assets Value (₹)",
                    min_value=0,
                    value=int(df["residential_assets_value"].median()),
                    step=100000,
                )
                commercial_assets = st.number_input(
                    "Commercial Assets Value (₹)",
                    min_value=0,
                    value=int(df["commercial_assets_value"].median()),
                    step=100000,
                )
                luxury_assets = st.number_input(
                    "Luxury Assets Value (₹)",
                    min_value=0,
                    value=int(df["luxury_assets_value"].median()),
                    step=100000,
                )
                bank_assets = st.number_input(
                    "Bank Assets Value (₹)",
                    min_value=0,
                    value=int(df["bank_asset_value"].median()),
                    step=100000,
                )

            submitted = st.form_submit_button(
                "Predict Loan Status",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            applicant = pd.DataFrame(
                [
                    {
                        "no_of_dependents": dependents,
                        "education": education,
                        "self_employed": self_employed,
                        "income_annum": income,
                        "loan_amount": loan_amount,
                        "loan_term": loan_term,
                        "cibil_score": cibil_score,
                        "residential_assets_value": residential_assets,
                        "commercial_assets_value": commercial_assets,
                        "luxury_assets_value": luxury_assets,
                        "bank_asset_value": bank_assets,
                    }
                ],
                columns=FEATURE_COLUMNS,
            )

            prediction = int(
                models[selected_model].predict(applicant)[0]
            )

            st.divider()

            if prediction == 1:
                st.success("✅ Model Result: Loan application predicted as APPROVED")
            else:
                st.error("❌ Model Result: Loan application predicted as REJECTED")

            st.caption(f"Model used: {selected_model}")
            st.warning(
                "This is an educational model prediction, not a real lending "
                "decision or financial recommendation. Actual loan decisions "
                "require regulated checks and human review."
            )

    # -------------------------------------------------------------------------
    #   Application footer
    # -------------------------------------------------------------------------

    st.divider()
    st.caption(
        "Educational project | Predictions are not a substitute for "
        "professional lending assessment."
    )


# =============================================================================
#   Application entry point
# =============================================================================

if __name__ == "__main__":
    main()
