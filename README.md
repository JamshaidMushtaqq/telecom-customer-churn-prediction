# 📊 Telecom Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to **churn (leave the service)** based on customer demographics, service usage, contract information, tenure, and billing details.

The project covers the complete Machine Learning workflow — from **Exploratory Data Analysis (EDA)** and preprocessing to model training, evaluation, and deployment using **Streamlit**.

---

## 🚀 Project Overview

Customer churn is an important business problem for telecom companies. Identifying customers who are likely to leave can help businesses understand customer behavior and take appropriate retention actions.

This project uses historical telecom customer data to build classification models that predict:

* **No Churn** — Customer is likely to stay
* **Churn** — Customer is likely to leave

The project also includes an interactive **Streamlit web application** where users can enter customer information and receive a churn prediction.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze customer churn patterns
* Perform exploratory data analysis
* Clean and preprocess the dataset
* Engineer meaningful features
* Train multiple Machine Learning classification models
* Compare model performance
* Evaluate models using multiple classification metrics
* Select a suitable model for prediction
* Deploy the trained model using Streamlit

---

## 📂 Project Structure

```text
Churn-prediction-ML-project/
│
├── Data/
│   └── telecom customer dataset
│
├── models/
│   ├── best_model_logistic_regression.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── reports/
│   └── project reports / evaluation outputs
│
├── src/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Dataset

The project uses a telecom customer churn dataset containing **7,043 customer records** and **21 features**.

The dataset contains information related to:

### Customer Information

* Gender
* Senior Citizen
* Partner
* Dependents

### Account Information

* Tenure
* Contract
* Payment Method
* Paperless Billing

### Services

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Billing

* Monthly Charges
* Total Charges

### Target Variable

```text
Churn
```

The target contains two classes:

```text
Yes → Customer churned
No  → Customer did not churn
```

---

## 🔍 Exploratory Data Analysis

The EDA phase was used to understand:

* Customer churn distribution
* Relationship between churn and contract type
* Tenure distribution
* Monthly charges
* Total charges
* Customer services
* Payment methods
* Demographic characteristics
* Potential patterns associated with churn

The dataset contains approximately:

* **73.46% No Churn**
* **26.54% Churn**

This class imbalance was considered when interpreting model performance.

---

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:

* Handling missing values
* Converting `TotalCharges` to numeric format
* Encoding categorical variables
* Feature scaling
* Preparing data for Machine Learning models

The project also uses a saved preprocessing object:

```text
models/preprocessor.pkl
```

This ensures that the Streamlit application can apply the same preprocessing used during model training.

---

## 🧠 Feature Engineering

Additional features were created to provide the models with potentially useful information.

### 1. Tenure Group

Customers were grouped according to their tenure:

```text
0–12 months
12–24 months
24–48 months
48–72 months
```

### 2. Number of Services

A feature was created to represent the number of services used by each customer.

### 3. Average Monthly Per Tenure

An additional charge-related feature was engineered using customer tenure and total charges.

Feature engineering was used to capture customer behavior that may not be directly represented by individual raw columns.

---

## 🤖 Machine Learning Models

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

The models were evaluated using a stratified train/test split.

```text
Training Set: 80%
Testing Set: 20%
```

---

## 📈 Model Evaluation

Multiple metrics were used instead of relying only on accuracy:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### Model Comparison

| Model                   |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **80.41%** | **66.12%** | **53.74%** | **59.29%** | **84.58%** |
| Random Forest           |     80.20% |     66.44% |     51.34% |     57.92% |     83.72% |
| Gradient Boosting       |     80.13% |     66.43% |     50.80% |     57.58% |     84.27% |
| XGBoost                 |     78.78% |     61.90% |     52.14% |     56.60% |     82.08% |
| Decision Tree           |     76.01% |     54.97% |     53.21% |     54.08% |     75.42% |

### Selected Model

Based on the evaluated test-set metrics, **Logistic Regression** was selected as the final model.

Test-set performance:

```text
Accuracy  : 80.41%
Precision : 66.12%
Recall    : 53.74%
F1-Score  : 59.29%
ROC-AUC   : 84.58%
```

> Note: Accuracy should not be interpreted as the percentage of churners correctly detected. Recall is particularly important for understanding how many actual churn customers are identified.

---

## 📊 Confusion Matrix

The selected Logistic Regression model produced approximately:

```text
                  Predicted
                  No     Yes

Actual No         932    103
Actual Yes        173    201
```

This means the model correctly identified **201 churn customers** in the test set while **173 actual churn customers were missed**.

This demonstrates why multiple evaluation metrics are important for a churn prediction problem.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

Users can enter customer information such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Contract
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Monthly Charges
* Total Charges
* Payment Method

The application then generates a churn prediction and probability.

### Run the application

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python -m streamlit run src/app.py
```

The application will open in your browser.

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Churn-prediction-ML-project.git
```

### 2. Navigate to the project

```bash
cd Churn-prediction-ML-project
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run src/app.py
```

---

## 🧰 Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Deployment

* Streamlit

### Model Persistence

* Pickle

---

## 📓 Notebooks

### `01_eda.ipynb`

Contains:

* Dataset exploration
* Data quality checks
* Missing-value analysis
* Churn distribution
* Visualizations
* Initial insights

### `02_preprocessing.ipynb`

Contains:

* Data cleaning
* Feature engineering
* Encoding
* Scaling
* Train/test preparation

### `03_model_training.ipynb`

Contains:

* Model training
* Model comparison
* Evaluation metrics
* Confusion matrix
* ROC-AUC analysis
* Final model selection

---

## 📌 Key Learnings

Through this project, I worked on the complete Machine Learning lifecycle:

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Model Deployment
```

One important learning from this project was that **accuracy alone is not sufficient** for evaluating a churn prediction model. Precision, recall, F1-score and ROC-AUC provide additional information about model performance.

---

## 🔮 Future Improvements

Potential improvements for future versions include:

* Cross-validation for more robust model evaluation
* Hyperparameter optimization
* Improved categorical encoding using pipelines
* Better handling of class imbalance
* Classification threshold optimization
* Probability calibration
* SHAP-based model explainability
* Improved feature engineering
* Automated model retraining
* Cloud deployment

---

## 👨‍💻 Author

**Jamshaid Mushtaq**

Aspiring Data Scientist / Machine Learning Enthusiast

If you found this project useful, feel free to ⭐ the repository.

---

## 📜 License

This project is created for educational and portfolio purposes.
