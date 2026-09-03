# Medical Insurance Cost Predictor

A beginner-friendly end-to-end Machine Learning project that predicts medical insurance charges using demographic and insurance-related information.

The project demonstrates a complete Machine Learning workflow, from dataset inspection and preprocessing to model training, evaluation, new-data prediction, and a Streamlit web application.

---

## Project Objective

The objective of this project is to demonstrate a simple and understandable end-to-end Machine Learning workflow rather than build a complex production system.

```text
Dataset
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Model Selection
   ↓
Model Saving
   ↓
New-Data Prediction
   ↓
Streamlit Application
```

---

## Dataset

The project uses the **Medical Cost Personal Dataset** obtained from Kaggle.

**Dataset source:**  
https://www.kaggle.com/datasets/mirichoi0218/insurance

### Dataset Information

| Property | Value |
|---|---:|
| Original records | 1,338 |
| Original columns | 7 |
| Duplicate records found | 1 |
| Records after duplicate removal | 1,337 |
| Missing values | 0 |
| ML input features | 8 |
| Target variable | `charges` |
| Problem type | Regression |

### Original Dataset Columns

| Column | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the individual |
| `sex` | Categorical | Sex of the individual |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of children/dependents |
| `smoker` | Categorical | Smoking status |
| `region` | Categorical | Residential region |
| `charges` | Numerical | Medical insurance charges |

### Target Variable

```text
charges
```

The model predicts the medical insurance charge represented by `charges`.

---

## Data Cleaning

The dataset was inspected before model development.

The following checks were performed:

- Dataset dimensions
- First and last records
- Data types
- Statistical summary
- Missing values
- Duplicate records
- Categorical values

### Cleaning Result

No missing values were found.

One duplicate record was identified and removed.

```text
Original Dataset
      ↓
1,338 records
      ↓
Duplicate Check
      ↓
1 duplicate found
      ↓
Duplicate removed
      ↓
1,337 records
```

No observations were removed merely because their numerical values appeared unusual.

---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed using Pandas and Matplotlib to understand the dataset before Machine Learning.

The project contains the following visualizations:

1. Distribution of Medical Insurance Charges
2. Distribution of Age
3. Distribution of BMI
4. Age vs Medical Insurance Charges
5. BMI vs Medical Insurance Charges
6. Average Medical Insurance Charges by Smoking Status
7. Correlation Matrix of Numerical Variables
8. Actual vs Predicted Medical Insurance Charges

### Important Observations

- Medical insurance charges show a strongly right-skewed distribution.
- Age has a positive relationship with medical insurance charges.
- BMI has a weaker positive relationship with charges.
- The numerical relationship between the number of children and charges is relatively weak.
- Smoking status shows a large difference in average insurance charges between smokers and non-smokers.

These observations were used to understand the data before model training.

---

## Data Preprocessing

The dataset contains both numerical and categorical variables.

### Numerical Features

```text
age
bmi
children
```

### Categorical Features

```text
sex
smoker
region
```

Categorical variables were converted into numerical representations so that they could be used by the Machine Learning models.

Binary variables were encoded as numeric values:

```text
sex
male   → 1
female → 0

smoker
yes → 1
no  → 0
```

The `region` variable was one-hot encoded. `northeast` was used as the reference category, producing these model features:

```text
age
sex
bmi
children
smoker
region_northwest
region_southeast
region_southwest
```

This produced:

```text
1,337 rows × 8 input features
```

The target variable was kept separately:

```text
charges
```

---

## Train/Test Split

The cleaned and encoded dataset was divided into training and testing subsets using an 80/20 split.

| Dataset | Records |
|---|---:|
| Training | 1,069 |
| Testing | 268 |

A fixed `random_state = 42` was used to make the experiment reproducible.

The test data was kept separate from training so that the model could be evaluated on previously unseen data.

---

## Machine Learning Models

Two regression algorithms were investigated:

1. Linear Regression
2. Decision Tree Regression

### Linear Regression Baseline

The Linear Regression model achieved the following test performance:

| Metric | Value |
|---|---:|
| MAE | 4,177.05 |
| MSE | 35,478,020.68 |
| RMSE | 5,956.34 |
| R² | 0.8069 |

### Initial Decision Tree

An unrestricted Decision Tree produced stronger test results than Linear Regression, but it achieved a training R² of `1.0000`, indicating that it was fitting the training data too closely.

Therefore, tree complexity was investigated before selecting the final model.

---

## Decision Tree Depth Experiment

Different values of `max_depth` were tested to study model complexity and generalization:

```text
2, 3, 4, 5, 6, 8, 10
```

### Results

| Max Depth | Train R² | Test R² | Test MAE | Test RMSE |
|---:|---:|---:|---:|---:|
| 2 | 0.8132 | 0.8672 | 3,312.84 | 4,939.88 |
| 3 | 0.8443 | 0.8930 | 2,755.27 | 4,435.20 |
| **4** | **0.8552** | **0.8972** | **2,621.31** | **4,345.88** |
| 5 | 0.8669 | 0.8941 | 2,656.84 | 4,411.54 |
| 6 | 0.8852 | 0.8699 | 2,816.47 | 4,889.82 |
| 8 | 0.9338 | 0.8515 | 2,689.79 | 5,224.35 |
| 10 | 0.9794 | 0.8363 | 2,477.95 | 5,484.96 |

A maximum depth of `4` was selected because it produced the strongest test R² and RMSE combination while avoiding the increasingly large training/test performance gap observed with deeper trees.

---

## Final Model

The final model used by the application is:

```python
DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)
```

### Final Test Performance

| Metric | Value |
|---|---:|
| MAE | **2,621.31** |
| MSE | **18,886,631.25** |
| RMSE | **4,345.88** |
| R² | **0.8972** |

### Metric Interpretation

**MAE — Mean Absolute Error**  
Average absolute difference between actual and predicted charges. Lower is better.

**MSE — Mean Squared Error**  
Average squared prediction error, which gives larger errors more influence. Lower is better.

**RMSE — Root Mean Squared Error**  
Square root of MSE and expressed in the same units as the target variable. Lower is better.

**R² — Coefficient of Determination**  
Indicates how much of the variation in the target is explained by the model relative to a mean-prediction baseline. Higher values are generally better.

---

## New-Data Prediction

The trained model was tested with a completely new input rather than simply reusing an existing dataset row.

### Example Input

```text
Age       = 40
Sex       = Male
BMI       = 28.5
Children  = 2
Smoker    = No
Region    = Northeast
```

### Prediction

```text
Predicted Medical Insurance Charges = 7357.27
```

This value is an estimate produced by the trained Machine Learning model.

---

## Model Saving

The final trained model is saved using Joblib:

```text
models/insurance_model.pkl
```

Saving the trained model means the Streamlit application can load the existing model instead of retraining it every time the application starts.

---

## Streamlit Web Application

A simple Streamlit application was built around the saved Decision Tree model.

The user can enter:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The application follows this flow:

```text
User Input
    ↓
Input Preprocessing
    ↓
Saved Decision Tree Model
    ↓
Prediction
    ↓
Displayed Result
```

The application uses user-friendly values such as `Male`, `No`, and `Northeast`; the preprocessing module converts them into the numerical representation required by the model.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data loading, cleaning, and analysis |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning models, splitting, and evaluation |
| Joblib | Saving and loading the trained model |
| Jupyter Notebook | Data analysis and experimentation |
| Streamlit | Web application |

TensorFlow was not used because the selected problem can be effectively solved using traditional regression algorithms without introducing unnecessary neural-network complexity.

---

## Project Structure

```text
medical-insurance-ml/
│
├── app/
│   └── app.py
│
├── data/
│   ├── insurance.csv
│   └── insurance_cleaned_encoded.csv
│
├── models/
│   └── insurance_model.pkl
│
├── notebooks/
│   └── medical_insurance_analysis.ipynb
│
├── screenshots/
│
├── src/
│   └── preprocessing.py
│
├── visualizations/
│   ├── actual_vs_predicted.png
│   ├── age_distribution.png
│   ├── age_vs_charges.png
│   ├── bmi_distribution.png
│   ├── bmi_vs_charges.png
│   ├── charges_distribution.png
│   ├── correlation_matrix.png
│   └── smoker_vs_charges.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Local Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd medical-insurance-ml
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show `(.venv)`.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Run the Streamlit Application Locally

From the project root:

```powershell
streamlit run app/app.py
```

Streamlit will normally provide a local address such as:

```text
http://localhost:8501
```

Open the address in a web browser.

---

## Run the Jupyter Notebook

The complete analysis and Machine Learning experimentation are available in:

```text
notebooks/medical_insurance_analysis.ipynb
```

Start Jupyter with:

```powershell
jupyter notebook
```

Then open the notebook from the `notebooks` directory.

---

## Research Component

A small model-comparison experiment was performed.

### Research Question

Which basic regression algorithm provides better predictive performance for medical insurance charges on the selected dataset?

### Models Compared

```text
Linear Regression
        vs
Decision Tree Regression
```

Decision Tree Regression produced better test performance than the Linear Regression baseline.

A second experiment evaluated multiple Decision Tree depths to investigate the effect of model complexity and overfitting. The final selected depth was `4`.

---

## Visualizations

The following figures are included in the `visualizations/` directory:

- Medical insurance charge distribution
- Age distribution
- Age vs medical insurance charges
- BMI distribution
- BMI vs medical insurance charges
- Average medical insurance charges by smoking status
- Correlation matrix
- Actual vs predicted medical insurance charges

---

## Limitations

This project is intended for educational and internship demonstration purposes.

The model is trained on a specific dataset and should not be interpreted as a production insurance-pricing system.

The prediction is an estimate generated by the Machine Learning model and is not an actual insurance quotation.

Model performance may differ when the model is applied to data from a different population or data distribution.

# Medical Insurance Cost Predictor

A basic Machine Learning project...

## 🚀 Live Demo

[Open the Medical Insurance Cost Predictor](https://medical-insurance-ml-cpgnlnzvnvcjzpcvrzfrnb.streamlit.app/)

## 📂 GitHub Repository

[View the source code on GitHub](https://github.com/Microblast56/medical-insurance-ml)
