# DSCI-441
# Eddie McGowan
# Diabetes Classification Model

## Project Description 
Over 800 million people globally are affected by diabetes, a condition that, if left unmanaged, can lead to serious, life-threatening complications. For this project, I developed ten classification models to predict the type of diabetes or prediabetic condition a patient may have, using both genetic markers and environmental factors. As a professional currently working in healthcare, my goal is for this model to eventually assist doctors in diagnosing a patient’s condition, enabling more effective preventive measures or targeted treatment strategies. I used a 70,000-record dataset from Kaggle for this project.

I will be evaluating the following models:

    - **Baseline Model** Decision Tree
    
    - **Baseline Model** Logistic Regression
    
    - **Baseline Model** K-Nearest Neighbors (KNN)
    
    - Linear Discriminant Analysis (LDA)
    
    - Support Vector Machine (SVM)
    
    - Random Forest
    
    - XGBoost
    
    - Custom Voting Ensemble
    
    - Custom Stacking
    
    - Multilayer Perceptron (Neural Network)

These models are evaluated based on predictive performance, execution time, and interpretability for non-technical audiences.

In addition, I am transitioning into a more data science–focused role at work. This project provides valuable hands-on experience in implementing and comparing machine learning models, allowing me to make more informed decisions in future model selection.

## Information About the Data Source

**Dataset:** [Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data)

Steps to download:
1. Follow the link.
2. Sign in to Kaggle.
3. Click "Download" in the top right corner.
4. In the dropdown menu, click "Download as ZIP."
5. Once downloaded, extract the ZIP file into your project folder.

This dataset has 34 columns and 70,000 rows. The columns are as follows:

| Column Name                      | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| **Target**                       | The type of diabetes or prediabetic condition diagnosed in the patient. |
| **Genetic Markers**               | Indicates whether specific genetic markers associated with diabetes are present ("Positive" or "Negative"). |
| **Autoantibodies**                | Presence of autoantibodies associated with autoimmune diabetes ("Positive" or "Negative"). |
| **Family History**                | Indicates whether the patient has a family history of diabetes ("True" or "False"). |
| **Environmental Factors**         | Notes environmental influences that may contribute to diabetes ("Present" or "Absent"). |
| **Insulin Levels**                | Insulin levels in the patient’s blood (µU/mL). |
| **Age**                           | Age of the patient at the time of data collection (years). |
| **BMI**                           | Body Mass Index of the patient. |
| **Physical Activity**             | The patient’s level of physical activity ("High," "Moderate," "Low"). |
| **Dietary Habits**                | The patient’s eating habits ("Healthy" or "Unhealthy"). |
| **Blood Pressure**                | Blood pressure levels (mmHg). |
| **Cholesterol Levels**            | Cholesterol levels (mg/dL). |
| **Waist Circumference**           | Waist circumference measurement (cm). |
| **Blood Glucose Levels**          | Blood glucose levels (mg/dL). |
| **Ethnicity**                     | Classification of diabetes risk based on ethnicity ("Low Risk" or "High Risk"). |
| **Socioeconomic Factors**         | Socioeconomic status classification ("Medium," "High," "Low"). |
| **Smoking Status**                | Whether the patient is a smoker ("Smoker" or "Non-Smoker"). |
| **Alcohol Consumption**           | Alcohol consumption levels ("Low," "Moderate," "High"). |
| **Glucose Tolerance Test**         | Result of glucose tolerance test ("Normal" or "Abnormal"). |
| **History of PCOS**               | Whether the patient has a history of Polycystic Ovary Syndrome (PCOS) ("True" or "False"). |
| **Previous Gestational Diabetes** | Whether the patient had gestational diabetes ("True" or "False"). |
| **Pregnancy History**             | Pregnancy outcome classification ("Normal" or "Complications"). |
| **Weight Gain During Pregnancy**  | Weight gained during pregnancy (kg). |
| **Pancreatic Health**             | Pancreatic function assessment. |
| **Pulmonary Function**            | Pulmonary function test results. |
| **Cystic Fibrosis Diagnosis**     | Whether the patient has cystic fibrosis ("True" or "False"). |
| **Steroid Use History**           | Whether the patient has a history of steroid use ("True" or "False"). |
| **Genetic Testing**               | Results of genetic testing ("Positive" or "Negative"). |
| **Neurological Assessments**      | Neurological test results. |
| **Liver Function Tests**          | Results of liver function tests ("Normal" or "Abnormal"). |
| **Digestive Enzyme Levels**       | Levels of digestive enzymes in the body. |
| **Urine Test**                    | Urine test results ("Protein Present," "Normal," or "Other"). |
| **Birth Weight**                  | The birth weight of the patient (grams). |
| **Early Onset Symptoms**          | Whether the patient exhibited early onset symptoms of diabetes ("True" or "False"). |

**Important:**  
- The dataset file name is `diabetes_dataset00.csv`.
- It should be placed inside the `data/` folder, which is one level below `DSCI 441 Milestone 2.ipynb`.

## List of packages required

The following Python packages are required to run this project:

- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scipy`  
- `numpy`  
- `scikit-learn`  
  - `GridSearchCV`, `StratifiedKFold`, `DecisionTreeClassifier`, `LogisticRegression`, `KNeighborsClassifier`,  
    `LinearDiscriminantAnalysis`, `LinearSVC`, `RandomForestClassifier`, `LabelEncoder`,  
    `accuracy_score`, `f1_score`, `make_pipeline`, `StandardScaler`, `StackingClassifier`
- `xgboost`  
  - `XGBClassifier`, `plot_importance`
- `tensorflow`  
  - `Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `Input`, `LeakyReLU`,  
    `to_categorical`, `EarlyStopping`
- `joblib`  
- `pickle`
   `steamlit`

You can install these dependencies using the following command:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost joblib tensorflow scipy

## How to Run the Code
Note, the steamlit application is already running on: http://magic02.cse.lehigh.edu:5006

To run the code:
1. Install all packages references above
2. Log into the Lehigh VPN
3. Enter the magic HPC http://magic02.cse.lehigh.edu
4. Upload dataset to the Jupyter server. (see instructions on how to download the data above)
5. Run DSCI 441 Milestone 2.ipynb. Model results are in the file. The output of this file is the trained models. Overall code will take a few hours to run.

To restart the streamlit application (need to run the above code first). Please email me if you run into any challenges.
1. (If needed) Kill any current process thats running
    - (lsof -i :5006) this will identify if any processes are running on the 5006 port.
    - (kill ####) replace #### with the ID where the process is running
2. Run streamlit application code/app by running on magic02 terminal "streamlit run aux_1.py --server.port 5006"
3. Open url http://magic02.cse.lehigh.edu:5006

## Project Plan
### **Goal:**
Identify the best-performing classification model for diabetes condition prediction.

### **Methodology**
5 fold cross valiation will be used for model evaluation. Initially, all features will be included. Grid search is  used to find the best parameters for each model

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

## ** Model Results**
- Computed accuracy, F1 score, training time, and testing time for each model
- Plotted decision tree to visualize how the model makes predictions.

### **Feature Importance Analysis:**
- Compared feature importance rankings from Decision Tree and Logistic Regression.
- Determined that **many of the same features** were influential across models.

## *Analysis/Conclusion*
- XGBoost: Highest accuracy and F1 score (~91%); fast training (93s); difficult to explain to non technical audiences
- Stacking Model: Comparable accuracy to XGBoost; training time longer (2 minutes); easier to explain compared to XGBoost
- Random Forest: Strong performance with balanced tradeoff between speed and interpretability
- MLP (Keras): Good performance, but below XGBoost; required one-hot encoding for labels
- All other models had comparatively poor accuracy and F1 Scores
- Across all models, the same key features consistently ranked at the top of the feature importance plots

## Resources & References
## Resources & References

### Dataset
- [Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data)

### Feature Engineering & Data Processing
- [Mapping True/False to Boolean](https://stackoverflow.com/questions/45196626/how-to-map-true-and-false-to-yes-and-no-in-a-pandas-data-frame-for-columns-o)
- [Printing Unique Values of Each Column](https://stackoverflow.com/questions/27241253/print-the-unique-values-in-every-column-in-a-pandas-dataframe)
- [Row Count Bar Chart](https://stackoverflow.com/questions/48939795/how-to-plot-a-count-bar-chart-grouping-by-one-categorical-column-and-coloring-by)
- [One Hot Encoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

### Statistical Analysis
- [Correlation Matrix](https://www.geeksforgeeks.org/create-a-correlation-matrix-using-python/)
- [KDE Plot](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)
- [Chi-Square Test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)

### Model Training & Evaluation
- [Time Module (Python)](https://docs.python.org/3/library/time.html)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html)
- [Decision Tree Feature Importance](https://stackoverflow.com/questions/69061767/how-to-plot-feature-importance-for-decisiontreeclassifier)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Logistic Regression Feature Importance](https://www.geeksforgeeks.org/understanding-feature-importance-in-logistic-regression-models/)
- [K-Nearest Neighbors (KNN)](https://scikit-learn.org/stable/modules/neighbors.html)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [LinearSVC (Support Vector Classifier)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/release_1.3.0/python/python_api.html)
- [Stacking Classifier (scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html#stacking)
- [Voting Classifier (scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
- [LDA (legacy scikit-learn)](https://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html)
- [Joblib Documentation](https://joblib.readthedocs.io/en/stable/)

### Neural Networks / Deep Learning
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Dropout Regularization Paper (Srivastava et al., 2014)](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Multilayer Perceptron Paper - Neural Networks](https://www.sciencedirect.com/science/article/pii/S0893608005800231?via%3Dihub)
- [World Scientific Paper on Diabetes Prediction](https://www.worldscientific.com/doi/abs/10.1142/S0218001411008683)
