# DSCI-441
# Eddie McGowan
# Diabetes Classification Model

## Overview
This project aims to develop a **classification model** to predict the **type of diabetes** (or prediabetic condition) a patient has, using a dataset of **70,000 records** from Kaggle. The model utilizes patient features such as **genetic markers, environmental factors, and lifestyle indicators** to predict diabetes conditions, allowing for **preventative and remediation strategies**.

## Project Plan
### **Goal:**
Identify the best-performing classification model for diabetes condition prediction.

### **Methodology**
An **80/20 train-test split** will be used for model evaluation. Initially, all features will be included. If computational constraints arise, **Principal Component Analysis (PCA)** will be applied for dimensionality reduction.

For the **baseline models**, standard configurations are used as found used in homeworks in this class. **Decision Tree:** Maximum depth of **5**. **K-Nearest Neighbors (KNN):** **n = 5**. For models developed in **Milestone 2**, hyperparameter tuning will be conducted using **randomized search** across a predefined range of values. See the models below.


### **Models:**
    - **Baseline Model** Logistic Regression
    - **Baseline Model** Decision Tree
    - **Baseline Model** K-Nearest Neighbors (KNN)
    - Linear Discriminant Analysis (LDA)
    - Support Vector Machine (SVM)
    - Random Forest
    - XGBoost
    - Multilayer Perceptron (Neural Network)

### **Evaluation Criteria:**
#### **Performance Metrics:**
- **Accuracy**
- **F1 Score**
- **Training Time**
- **Testing Time**

#### **Model Interpretability:**
- Example: "If a patient has a history of smoking and a BMI of 38, they are at risk of developing Type 2 Diabetes."

## Dataset & Preprocessing
### **Exploratory Data Analysis (EDA):**
1. **Checked for duplicates:** None found.
2. **Checked and adjusted data types:**
   - Converted categorical columns to category type
   - Converted binary features to Boolean (True/False)
3. **One-hot encoding of categorical features.**
4. **Summary statistics & feature distributions:**
   - Identified no missing values or outliers.
   - Verified class balance in the target variable.
5. **Created a correlation matrix of all the features:**
   - Retained some features with correlations **> 0.7** after analyzing the correlated features.

## Statistical Analysis
### **Hypothesis Testing:**
- **Question:** Does high BMI (≥30) correlate with a higher likelihood of Type 2 Diabetes?
- **Test:** **Chi-Square Test**
- **Results:** Rejected null hypothesis → High BMI patients are more likely to have Type 2 Diabetes.

### **Feature Distributions:**
- KDE plots were generated for all numerical features to analyze distributions.
- Identified some skewed distributions and bimodal distributions (e.g., "Weight Gain During Pregnancy").

## **Baseline Model Results**
### **Decision Tree Insights:**
- Plotted decision tree to visualize how the model makes predictions.

### **Feature Importance Analysis:**
- Compared feature importance rankings from Decision Tree and Logistic Regression.
- Determined that **many of the same features** were influential across models.

## **Next Steps**
- Implement **remaining models**:
  - Linear Discriminant Analysis (LDA)
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
  - Multilayer Perceptron (Neural Network using TensorFlow/Keras)
- Optimize model parameters for better performance.
- Apply **Principal Component Analysis (PCA)** for dimensionality reduction if needed.

## Resources & References
- **Dataset:** [Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data)
- **Feature Engineering & Data Processing:**
  - [Mapping True/False to Boolean](https://stackoverflow.com/questions/45196626/how-to-map-true-and-false-to-yes-and-no-in-a-pandas-data-frame-for-columns-o)
  - [Printing Unique Values of Each Column](https://stackoverflow.com/questions/27241253/print-the-unique-values-in-every-column-in-a-pandas-dataframe)
  - [Row Count Bar Chart](https://stackoverflow.com/questions/48939795/how-to-plot-a-count-bar-chart-grouping-by-one-categorical-column-and-coloring-by)
  - [One Hot Encoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- **Statistical Analysis:**
  - [Correlation Matrix](https://www.geeksforgeeks.org/create-a-correlation-matrix-using-python/)
  - [KDE Plot](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)
  - [Chi-Square Test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
- **Model Training & Evaluation:**
  - [Time Module](https://docs.python.org/3/library/time.html)
  - [Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
  - [Decision Tree Feature Importance](https://stackoverflow.com/questions/69061767/how-to-plot-feature-importance-for-decisiontreeclassifier)
  - [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  - [Logistic Regression Feature Importance](https://www.geeksforgeeks.org/understanding-feature-importance-in-logistic-regression-models/)
  - [K-Nearest Neighbors (KNN)](https://scikit-learn.org/stable/modules/neighbors.html)
