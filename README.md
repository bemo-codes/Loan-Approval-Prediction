# 🏦 Loan Approval Prediction - Ensemble ML Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Ensemble Learning](https://img.shields.io/badge/Ensemble-Voting-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

An interactive Streamlit application that explores loan applicant information, compares classification models, and predicts whether a loan application is approved or rejected based on patterns in the selected dataset.

## ✨ Features

- Loan dataset preview and descriptive statistics
- Approval and rejection distribution
- Applicant feature correlation analysis
- Approval-rate insights by education and employment type
- Comparison of Decision Tree, Logistic Regression, KNN, and Voting Classifier
- Accuracy, precision, recall, and F1-score evaluation
- Confusion matrix and classification report
- Interactive loan application prediction form

## 🤖 Machine Learning Approach

The project follows the original assignment's model design and extends it to support a larger, mixed-type dataset.

| Model | Purpose |
|---|---|
| Decision Tree | Learns rule-based splits in applicant data |
| Logistic Regression | Provides a linear classification baseline |
| K-Nearest Neighbors | Classifies applications using nearby examples |
| Voting Classifier | Combines the predictions of the three models |

The Voting Classifier uses **hard voting**, as in the original assignment: the final class is selected by majority vote.

### Preprocessing

The larger dataset contains numeric and categorical fields. The application handles them with a scikit-learn `ColumnTransformer`:

- Numeric fields: median imputation and standardization.
- Categorical fields: most-frequent imputation and one-hot encoding.
- Unknown categorical values at prediction time are ignored safely by the encoder.
- `loan_id` is removed because it is a row identifier, not a useful applicant feature.

Preprocessing is included inside each model pipeline. This ensures that the scaler and encoders are fitted on the training data and then reused to transform the test data. It also prevents the train/test scaling issue present in the original assignment code.

## 📊 Dataset

### Recommended dataset

**Loan Approval Prediction Dataset** — 4,269 loan applications and 13 columns (including the row ID and target).

- Kaggle source: [Loan Approval Prediction Dataset by Archit Sharma](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- GitHub CSV mirror: [loan_approval_dataset.csv](https://github.com/michaelmml/Financial-Data-Science/blob/main/loan_approval_dataset.csv)

The application expects the downloaded file to be named exactly:

```text
loan_approval_dataset.csv
```

### Dataset features

| Feature | Description |
|---|---|
| `loan_id` | Unique application identifier; dropped before training |
| `no_of_dependents` | Number of applicant dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Applicant's annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term in years |
| `cibil_score` | Applicant's CIBIL score |
| `residential_assets_value` | Residential asset value |
| `commercial_assets_value` | Commercial asset value |
| `luxury_assets_value` | Luxury asset value |
| `bank_asset_value` | Bank asset value |
| `loan_status` | Target: Approved / Rejected |

The original assignment file `Customer_Loan_Approval.csv` contains only 50 rows and seven columns. The larger dataset is used for this project because it offers more applicant records and additional categorical and financial attributes.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive dashboard |
| Pandas | Dataset loading and processing |
| NumPy | Numerical operations |
| Matplotlib | Charts |
| Seaborn | Statistical visualizations |
| Scikit-learn | Preprocessing, classifiers, voting and evaluation |

## 🖥️ Dashboard Sections

1. **Dataset Overview** – Inspect applicant records, target distribution, descriptive statistics and missing values.
2. **Applicant Insights** – Explore numerical correlations, approval rates by education and employment type, and CIBIL-score distributions.
3. **Model Comparison** – Compare all four models using accuracy, precision, recall and F1-score; inspect confusion matrices and classification reports.
4. **Loan Predictor** – Enter a sample applicant profile and select a trained model to generate an approval/rejection prediction.

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── app.py
├── loan_approval_dataset.csv
├── download_dataset.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
└── screenshots/
```

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 or newer
- pip
- Internet connection for the optional dataset downloader

### 1. Clone your repository

```bash
git clone https://github.com/bemo-codes/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

Or download the ZIP and open the extracted folder in VS Code.

### 2. Create and activate a virtual environment

**Windows Command Prompt / PowerShell:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Git Bash:**

```bash
python -m venv venv
source venv/Scripts/activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

You can download the dataset manually from Kaggle or GitHub.

**Manual GitHub download:**

1. Open the [CSV file on GitHub](https://github.com/michaelmml/Financial-Data-Science/blob/main/loan_approval_dataset.csv).
2. Click **Raw** or **Download raw file**.
3. Save it in the same folder as `app.py`.
4. Rename it to `loan_approval_dataset.csv` if required.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

## ⚠️ Limitations

- The dataset contains 4,269 applications and may not represent all lenders, countries, or applicant populations.
- Model metrics are specific to the supplied dataset and chosen train/test split.
- The Voting Classifier uses hard voting, so all three models contribute one class vote.
- The current project is an educational demonstration, not a real lending or credit-scoring system.
- Loan approval decisions are high-impact decisions and should not be made solely using this model. Real systems require careful validation, fairness assessment, regulatory review, explainability, and human oversight.
- Review the dataset's licence and terms before redistributing the CSV in a public repository.

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**  
GitHub: [@bemo-codes](https://github.com/bemo-codes)

## 📄 License

This project is distributed under the MIT License. See `LICENSE` for details.

---

<p align="center">Built with Python 🐍, Streamlit ⚡ and Scikit-learn 🤖</p>
