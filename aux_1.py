import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier

# --- Other tools ---
from scipy.stats import mode


from scipy.stats import mode

class CustomVotingModel:
    def __init__(self, model_dict, model_names):
        self.model_dict = model_dict
        self.model_names = model_names

    def predict(self, X):
        preds = np.array([
            self.model_dict[name].predict(X)
            for name in self.model_names
        ])
        return mode(preds, axis=0).mode.squeeze()


# Load model metrics from pickle
with open('model_metrics.pkl', 'rb') as f:
    model_metrics = pickle.load(f)

# Convert to DataFrame
metrics_df = pd.DataFrame(model_metrics).T

# Set page config
st.set_page_config(page_title="Diabetes Model Comparison", layout="wide")

# Section 1: Background & Goal
st.title("🧠 Diabetes Prediction Models")
st.header("1. Background & Goal")
st.markdown(
    """
    Over 800 million people globally are affected by diabetes, a condition that if left unmanaged—can lead to serious, life threatening complications. For my project, I’m developing a classification model that predicts the type of diabetes or prediabetic condition a patient may have, based on both genetic markers and environmental factors. Since I currently work in healthcare, my goal is for this model to eventually serve as a tool to help doctors apply preventive measures or targeted treatment strategies more effectively.

While this project is currently focused on diabetic patients, it serves as a proof of concept. The same approach could easily be extended to other chronic conditions such as arthritis or Alzheimer’s, where early detection plays a key role in improving outcomes.

The dataset I’m using comes from Kaggle and includes over 70,000 patient records, offering a strong foundation for building and validating the model.
    [Diabetes Dataset](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data)
    """
)
# Section 2: Models Tested

# Define base estimators for ensembles
import joblib
from tensorflow.keras.models import load_model

models = {
    "Baseline: Decision Tree": joblib.load("tree_model.pkl"),
    "Baseline: Logistic Regression": joblib.load("logistic_model.pkl"),
    "Baseline: KNN": joblib.load("knn_model.pkl"),
    "LDA": joblib.load("lda_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "XGBoost": joblib.load("xgb_model.pkl"),
    "Custom: Stacking": joblib.load("stacking_model.pkl"),
    "Multilayer Perceptron": load_model("mlp_model.h5")  # Keras model
}

voting_model = CustomVotingModel(
    model_dict=models,  # Use the dictionary you already have
    model_names=["LDA", "SVM", "Baseline: Decision Tree", "Baseline: KNN", "Baseline: Logistic Regression", "Random Forest"]
)
models["Custom: Voting"] = voting_model

label_encoder = joblib.load("label_encoder.pkl")

st.header("2. Models Tested")
model_list = sorted(model_metrics.keys())
st.write("The following models were evaluated:")
st.markdown("\n".join([f"- {model}" for model in model_list]))

# Section 3: Results
st.header("3. Results")

def plot_bar_chart(data, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    data.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    st.pyplot(fig)

# Create two rows of columns (2x2 grid)
col1, col2 = st.columns(2)
with col1:
    plot_bar_chart(metrics_df["accuracy"], "Model Accuracy", "Accuracy")

with col2:
    plot_bar_chart(metrics_df["f1"], "Model F1 Score", "F1 Score")

col3, col4 = st.columns(2)
with col3:
    plot_bar_chart(metrics_df["train_time"], "Training Time by Model", "Seconds")

with col4:
    plot_bar_chart(metrics_df["test_time"], "Testing Time by Model", "Seconds")


# Section 4: Full Comparison Table
st.header("4. Full Comparison Table")

# Round metric_df columns and reset index for merge
# Convert accuracy and f1 to percentages, then round and cast to int
metrics_rounded = metrics_df.copy()
metrics_rounded["accuracy"] = (metrics_rounded["accuracy"] * 100).round(0).astype(int)
metrics_rounded["f1"] = (metrics_rounded["f1"] * 100).round(0).astype(int)

metrics_rounded = metrics_rounded[["accuracy", "f1", "train_time", "test_time"]].reset_index().rename(columns={
    "index": "Model",
    "accuracy": "Accuracy (%)",
    "f1": "F1 Score (%)",
    "train_time": "Training Time (s)",
    "test_time": "Testing Time (s)"
})


# Create additional columns
extra_data = {
    "Model": [
        "Baseline: Decision Tree", "Baseline: Logistic Regression", "Baseline: KNN", "LDA", "SVM",
        "Random Forest", "XGBoost", "Multilayer Perceptron", "Custom: Ensemble Voting", "Custom: Stacking"
    ],
    "Interpretability": ["Great", "Great", "Good", "Average", "Average", "Average", "Poor", "Poor", "Average", "Average"],
    "Epochs or Trees": ["NA", "100 (default)", "NA", "NA", "55", "100", "100", "55", "NA", "1000"]
}

extra_df = pd.DataFrame(extra_data)

# Merge on Model
full_df = pd.merge(metrics_rounded, extra_df, on="Model", how="left")

# Display in Streamlit
#st.dataframe(full_df, use_container_width=True)
st.dataframe(full_df)

# Section 5: Conclusion
st.header("5. Conclusion")
st.markdown("""
I built 10 models.

- **Best Model for Performance**: XGBoost
  - Tied for top performance with Stacking model
  - Significantly faster for training (3 seconds vs 2 minutes for Stacking)
  - However, it's harder to explain in layman's terms

- **Best Model for Implementation**: Stacking model
  - Easier to explain than XGBoost, while maintaining similar performance
  - Longer training time has minimal impact since training is infrequent
  - Testing time is similar (under 1 second)
""")

# Load model and label encoder
st.header("4. Predict My Condition")
st.markdown("Use the dropdowns below to select patient features and predict the diabetes condition.")


# Load base dataset for value reference (used just for UI dropdown options)
diabetes = pd.read_csv("results.csv")

# All columns except Target
input_fields = [col for col in diabetes.columns if col != "Target"]

# Build the UI
user_input = {}
for i in range(0, len(input_fields), 4):
    cols = st.columns(4)
    for j, field in enumerate(input_fields[i:i+4]):
        with cols[j]:
            user_input[field] = st.selectbox(field, sorted(diabetes[field].dropna().unique()))

# Create raw input DataFrame
user_df_raw = pd.DataFrame([user_input])

# Append user input to the base data for preprocessing
diabetes = pd.concat([diabetes, user_df_raw])


# Map booleans
bool_map = {"Yes": True, "No": False}
diabetes["Family History"] = diabetes["Family History"].map(bool_map)
diabetes["History of PCOS"] = diabetes["History of PCOS"].map(bool_map)
diabetes["Previous Gestational Diabetes"] = diabetes["Previous Gestational Diabetes"].map(bool_map)
diabetes["Cystic Fibrosis Diagnosis"] = diabetes["Cystic Fibrosis Diagnosis"].map(bool_map)
diabetes["Steroid Use History"] = diabetes["Steroid Use History"].map(bool_map)
diabetes["Early Onset Symptoms"] = diabetes["Early Onset Symptoms"].map(bool_map)

# Map binary feature transformations
diabetes["Genetic Markers Positive"] = diabetes["Genetic Markers"].map({"Positive": True, "Negative": False}).fillna(False)
diabetes.drop(columns=["Genetic Markers"], inplace=True)

diabetes["Autoantibodies Positive"] = diabetes["Autoantibodies"].map({"Positive": True, "Negative": False}).fillna(False)
diabetes.drop(columns=["Autoantibodies"], inplace=True)

diabetes["Environmental Factors Present"] = diabetes["Environmental Factors"].map({"Present": True, "Absent": False}).fillna(False)
diabetes.drop(columns=["Environmental Factors"], inplace=True)

diabetes["Dietary Habits Unhealthy"] = diabetes["Dietary Habits"].map({"Unhealthy": True, "Healthy": False}).fillna(False)
diabetes.drop(columns=["Dietary Habits"], inplace=True)

diabetes["Smoking Status Smoker"] = diabetes["Smoking Status"].map({"Smoker": True, "Non-Smoker": False}).fillna(False)
diabetes.drop(columns=["Smoking Status"], inplace=True)

diabetes["Glucose Tolerance Test Abnormal"] = diabetes["Glucose Tolerance Test"].map({"Abnormal": True, "Normal": False}).fillna(False)
diabetes.drop(columns=["Glucose Tolerance Test"], inplace=True)

diabetes["Pregnancy History Complications"] = diabetes["Pregnancy History"].map({"Complications": True, "Normal": False}).fillna(False)
diabetes.drop(columns=["Pregnancy History"], inplace=True)

diabetes["Genetic Testing Positive"] = diabetes["Genetic Testing"].map({"Positive": True, "Negative": False}).fillna(False)
diabetes.drop(columns=["Genetic Testing"], inplace=True)

diabetes["Liver Function Tests Abnormal"] = diabetes["Liver Function Tests"].map({"Abnormal": True, "Normal": False}).fillna(False)
diabetes.drop(columns=["Liver Function Tests"], inplace=True)

diabetes["Ethnicity High Risk"] = diabetes["Ethnicity"].map({"High Risk": True, "Low Risk": False}).fillna(False)
diabetes.drop(columns=["Ethnicity"], inplace=True)

# One-hot encode
categorical_columns = ["Physical Activity", "Socioeconomic Factors", "Alcohol Consumption", "Urine Test"]
diabetes_encoded = pd.get_dummies(diabetes, columns=categorical_columns)

# Predict
expected_columns = joblib.load("training_columns.pkl")

input_row = diabetes_encoded.iloc[[-1]]  # 2D input
if "Obese" in expected_columns and "BMI" in diabetes_encoded.columns:
    diabetes_encoded["Obese"] = diabetes_encoded["BMI"] >= 30

# Remove target if it slipped in
if "Target" in diabetes_encoded.columns:
    diabetes_encoded_pred = diabetes_encoded.drop(columns=["Target"])

# Reindex to expected columns and fill missing with 0
input_row = diabetes_encoded_pred.iloc[[-1]].reindex(columns=expected_columns, fill_value=0)

# Sort model names alphabetically
model_names_sorted = sorted(models.keys())

selected_model_name = st.selectbox("Select a model for prediction", model_names_sorted, key="model_selector")
selected_model = models[selected_model_name]

# Special case for MLP (requires scaling)
if selected_model_name == "Multilayer Perceptron":
    scaler = joblib.load("scaler.pkl")
    input_row_scaled = scaler.transform(input_row)
    pred_probs = selected_model.predict(input_row_scaled)
    y_pred = np.argmax(pred_probs, axis=1)
else:
    y_pred = selected_model.predict(input_row)

# ✅ Now decode and display the prediction
label_encoder = joblib.load("label_encoder.pkl")
predicted_label = label_encoder.inverse_transform(y_pred)[0]
st.success(f"🎯 Predicted Condition: {predicted_label}")





# Section 5: Results Filtered by Target
st.header("🎯 Evaluate Pretrained Models on a Selected Class")

# Load the label encoder used during training

# Choose a class
available_classes = diabetes_encoded["Target"].dropna().astype(str).unique()
selected_class = st.selectbox("Select class to evaluate", sorted(available_classes))

# Filter data for just that class
filtered_df = diabetes_encoded[diabetes_encoded["Target"] == selected_class]
X_filtered = filtered_df.drop(columns=["Target"])
y_true = filtered_df["Target"]

# Encode target if needed
y_true_enc = label_encoder.transform(y_true)

# Initialize storage
model_names = [name for name in models]
accuracy_scores = []
f1_scores = []

for name in model_names:
    model = models[name]

    if name == "Multilayer Perceptron":
        scaler = joblib.load("scaler.pkl")
        X_input = scaler.transform(X_filtered)
        pred_probs = model.predict(X_input)
        y_pred_enc = np.argmax(pred_probs, axis=1)
    else:
        y_pred_enc = model.predict(X_filtered)

    # Decode if needed
    y_pred_enc = np.atleast_1d(y_pred_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)

# Plot results
# Create dataframes from scores
acc_df = pd.DataFrame({"Model": model_names, "Accuracy": accuracy_scores})
f1_df = pd.DataFrame({"Model": model_names, "F1 Score": f1_scores})

# Optional: Clean up long names for better x-axis alignment
acc_df["Model"] = acc_df["Model"].str.replace("Baseline: ", "").str.replace("Multilayer Perceptron", "MLP")
f1_df["Model"] = f1_df["Model"].str.replace("Baseline: ", "").str.replace("Multilayer Perceptron", "MLP")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
# Sort by Model name alphabetically
acc_df = acc_df.sort_values("Model")
f1_df = f1_df.sort_values("Model")

# Accuracy plot
sns.barplot(data=acc_df, x="Model", y="Accuracy", palette="Blues", ax=axes[0])
axes[0].set_title(f"Accuracy\non '{selected_class}'")
axes[0].set_xticklabels(acc_df["Model"], rotation=45, ha="right")
axes[0].set_ylim(0, 1)

# F1 plot
sns.barplot(data=f1_df, x="Model", y="F1 Score", palette="Oranges", ax=axes[1])
axes[1].set_title(f"F1 Score\non '{selected_class}'")
axes[1].set_xticklabels(f1_df["Model"], rotation=45, ha="right")
axes[1].set_ylim(0, 1)

st.pyplot(fig)


# Section 6: References
st.header("6. References")
st.markdown("""
- Dataset: https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data
- Diabetes = https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444#:~:text=Long%2Dterm%20complications%20of%20diabetes,lead%20to%20type%202%20diabetes.
- Map True and False to Boolean: https://stackoverflow.com/questions/45196626/how-to-map-true-and-false-to-yes-and-no-in-a-pandas-data-frame-for-columns-o
- Print Unique Values: https://stackoverflow.com/questions/27241253/print-the-unique-values-in-every-column-in-a-pandas-dataframe
- Row Count Bar Chart: https://stackoverflow.com/questions/48939795/how-to-plot-a-count-bar-chart-grouping-by-one-categorical-column-and-coloring-by
- One Hot Encoding: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
- Correlation Matrix: https://www.geeksforgeeks.org/create-a-correlation-matrix-using-python/
- KDE Plot: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
- Chi Square Test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
- Time Library: https://docs.python.org/3/library/time.html
- Decision Tree: https://scikit-learn.org/stable/modules/tree.html
- DT Feature Importance: https://stackoverflow.com/questions/69061767/how-to-plot-feature-importance-for-decisiontreeclassifier
- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- LR Feature Importance: https://www.geeksforgeeks.org/understanding-feature-importance-in-logistic-regression-models/
- KNN: https://scikit-learn.org/stable/modules/neighbors.html
- LDA: https://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
- SVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
- Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- XGBoost: https://xgboost.readthedocs.io/en/release_1.3.0/python/python_api.html
- Model Approach: https://www.worldscientific.com/doi/abs/10.1142/S0218001411008683
- Voting Classifier: https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
- Stacking Model: https://www.sciencedirect.com/science/article/pii/S0893608005800231?via%3Dihub
- Stacking Impl.: https://scikit-learn.org/stable/modules/ensemble.html#stacking
- TensorFlow Keras: https://www.tensorflow.org/guide/keras
- Dropout Paper: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
""")
