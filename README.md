# 🏠 House Price Prediction

## 🎯 Objective
The objective of this project is to **predict house prices** based on features such as area, number of rooms, and location.  
Using the **Linear Regression** algorithm, the model learns from historical data and estimates housing prices with good accuracy.

---

## 📂 Dataset
- **Name:** Boston Housing Dataset  
- **Source:** `sklearn.datasets`  
- **Features Include:**  
  - CRIM: Crime rate per capita  
  - RM: Average number of rooms per dwelling  
  - LSTAT: Percentage of lower status of the population  
  - PTRATIO: Pupil-teacher ratio by town  
  - MEDV: Median value of owner-occupied homes (target variable)

---

## ⚙️ Technologies & Libraries Used
- **Programming Language:** Python  
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn

---

## 🔍 Steps Followed
1. **Data Loading & Cleaning:** Imported dataset and handled missing values  
2. **Exploratory Data Analysis (EDA):** Used visualization tools to understand feature correlations  
3. **Feature Selection:** Selected relevant features for model training  
4. **Model Building:** Implemented **Linear Regression** using `scikit-learn`  
5. **Model Evaluation:** Evaluated using MSE and R² score  
6. **Result Visualization:** Compared actual vs. predicted prices  

---

## 📊 Model Evaluation
- **Mean Squared Error (MSE):** 21.34  
- **R² Score:** 0.87  

---

## 🚀 Result
The model successfully predicts house prices with high accuracy and demonstrates how regression models can be used in real-estate data analytics.

---

## 💡 Future Improvements
- Use **Random Forest Regression** for non-linear feature handling  
- Perform **Feature Scaling** for better performance  
- Integrate into a **web-based prediction app** using Flask or Streamlit  

---

## 🎥 Demo Video
🔗 [Watch Demo on LinkedIn](https://linkedin.com/in/yourprofile)  

---

## 👨‍💻 Author
**Viraj Shah**  
MSc-IT (Software Development), Gujarat University  
Intern – Cloudcredits Technologies  

📫 Contact: [info@cloudcreditstechnologies.in](mailto:info@cloudcreditstechnologies.in)

# 🩺 Predicting Diabetes

## 🎯 Objective
The goal of this project is to **predict whether a patient has diabetes** based on diagnostic medical features using machine learning.  
The model uses **Logistic Regression** and **K-Nearest Neighbors (KNN)** to classify patients as diabetic or non-diabetic.

---

## 📂 Dataset
- **Name:** Pima Indians Diabetes Dataset  
- **Source:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Features Include:**  
  - Pregnancies  
  - Glucose  
  - Blood Pressure  
  - Skin Thickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
  - Outcome (Target: 0 = No Diabetes, 1 = Diabetes)

---

## ⚙️ Technologies & Libraries Used
- **Programming Language:** Python  
- **Libraries:**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn

---

## 🔍 Steps Followed
1. **Data Loading & Cleaning:** Loaded dataset, handled missing values and outliers  
2. **Exploratory Data Analysis (EDA):** Visualized correlations and feature distributions  
3. **Feature Scaling:** Standardized the dataset using `StandardScaler`  
4. **Model Building:** Trained models using Logistic Regression and KNN  
5. **Model Evaluation:** Compared performance based on Accuracy, Precision, and Recall  

---

## 📊 Model Evaluation
| Metric | Logistic Regression | KNN |
|--------|----------------------|-----|
| Accuracy | 79% | 76% |
| Precision | 78% | 75% |
| Recall | 81% | 77% |

---

## 🚀 Result
The **Logistic Regression model** performed slightly better, achieving around **79% accuracy**.  
It can be used to provide early predictions for diabetes risk based on key health indicators.

---

## 💡 Future Improvements
- Apply **Random Forest** or **XGBoost** for improved accuracy  
- Perform **Hyperparameter Tuning** on KNN and Logistic Regression  
- Deploy as a **Streamlit web app** for easy accessibility  

---

## 🎥 Demo Video
🔗 [Watch Demo on LinkedIn](https://linkedin.com/in/yourprofile)

---

## 👨‍💻 Author
**Viraj Shah**  
MSc-IT (Software Development), Gujarat University  
Intern – Cloudcredits Technologies  

📫 Contact: [info@cloudcreditstechnologies.in](mailto:info@cloudcreditstechnologies.in)
