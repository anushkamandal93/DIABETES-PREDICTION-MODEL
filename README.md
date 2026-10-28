[README.md](https://github.com/user-attachments/files/34072429/README.md)
# 🩺 Diabetic Prediction Model

A machine learning project that predicts the likelihood of diabetes from a patient's health details. It includes a Jupyter notebook for data exploration and model building, and a Streamlit web app for making predictions interactively.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is not a medical tool and must not be used for diagnosis. Please consult a qualified doctor for medical advice.

---

## 📂 Project Structure

```
Diabetes Prediction P2/
├── Diabetes_prediction_ML.ipynb      # Data exploration and model building
├── app.py                            # Streamlit web app
├── requirements.txt                  # Python dependencies
├── diabetes_prediction_dataset.csv   # Dataset (see below)
├── diabetes_pipeline.pkl             # Trained model (created on first run of app.py)
└── README.md
```

## 📊 Dataset

- **Source:** Kaggle - [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (`iammustafatz/diabetes-prediction-dataset`)
- **Target column:** `diabetes` (0 = no diabetes, 1 = diabetes)

| Feature | Description |
|---|---|
| `gender` | Female / Male / Other |
| `age` | Age in years |
| `hypertension` | 0 = no, 1 = yes |
| `heart_disease` | 0 = no, 1 = yes |
| `smoking_history` | never, No Info, current, former, ever, not current |
| `bmi` | Body Mass Index |
| `HbA1c_level` | Average blood sugar level over about 3 months (%) |
| `blood_glucose_level` | Current blood glucose level (mg/dL) |

The dataset is heavily imbalanced: far fewer diabetic than non-diabetic records. The model uses balanced class weights to handle this, and recall for the diabetic class is reported alongside accuracy.

## ⚙️ Model

- **Preprocessing:** standard scaling for numeric columns, one-hot encoding for categorical columns.
- **Algorithm:** Random Forest Classifier inside a scikit-learn `Pipeline`.
- **Split:** 80% training / 20% testing (stratified).

## 🚀 Getting Started

### 1. Clone or download the project

Place all project files in one folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For the notebook, you also need:

```bash
pip install seaborn matplotlib "kagglehub[pandas-datasets]"
```

### 3. Get the dataset

Download it from Kaggle and place `diabetes_prediction_dataset.csv` in the project folder. Or, from the notebook:

```python
import os
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
df = pd.read_csv(os.path.join(path, "diabetes_prediction_dataset.csv"))
df.to_csv("diabetes_prediction_dataset.csv", index=False)
```

### 4. Run the app

```bash
python -m streamlit run app.py
```

The first launch trains the model and saves it as `diabetes_pipeline.pkl`, which can take a minute. After that, the app opens in your browser at `http://localhost:8501`.

### 5. Run the notebook

Open `Diabetes_prediction_ML.ipynb` in VS Code or Jupyter, select the Python environment where the packages are installed, and click **Run All**.

## 🖥️ Using the App

1. Enter the patient's gender, age, BMI, smoking history, HbA1c level, blood glucose level, and whether they have hypertension or heart disease.
2. Click **Predict**.
3. The app shows a low or high risk result along with the predicted probability.

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| Yellow underlines under imports in VS Code | Select the correct Python interpreter (Ctrl+Shift+P, then *Python: Select Interpreter*) and kernel, then reload the window |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the same Python environment |
| `streamlit` command not found | Use `python -m streamlit run app.py` |
| `kagglehub.load_dataset` error | Use `kagglehub.dataset_download(...)` with `pd.read_csv`, as shown above |
| App says dataset not found | Put `diabetes_prediction_dataset.csv` in the same folder as `app.py` |

## 🧰 Tech Stack

Python 3.13 · pandas · NumPy · scikit-learn · seaborn · matplotlib · Streamlit · joblib · kagglehub

## 🔮 Future Improvements

- Compare more models (Logistic Regression, XGBoost) and tune hyperparameters.
- Add feature importance and explainability charts to the app.
- Deploy the app on Streamlit Community Cloud.

## 👩‍💻 Author

Anushka

## 📜 License

Released for educational use. The dataset belongs to its original author on Kaggle.
