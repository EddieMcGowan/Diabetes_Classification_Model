import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image

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
    For my project, I am creating a classification model to predict the type of diabetes or prediabetic condition
    a patient has based on their features, including genetic markers and environmental factors. I currently work in health care,
    and the goal of this model is to help doctors apply preventative or remediation strategies.

    A good model should have high performance and be explainable in layman's terms so that both doctors and patients can
    understand the reasoning behind each prediction. This dataset was sourced from Kaggle and includes over 70,000 patient records:
    [Diabetes Dataset](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data)
    """
)

# Section 2: Models Tested
st.header("2. Models Tested")
model_list = list(model_metrics.keys())
st.write("The following models were evaluated:")
st.markdown("\n".join([f"- {model}" for model in model_list]))

# Section 3: Results
st.header("3. Results")

def plot_bar_chart(data, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    data.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    st.pyplot(fig)

plot_bar_chart(metrics_df["accuracy"], "Model Accuracy", "Accuracy")
plot_bar_chart(metrics_df["f1"], "Model F1 Score", "F1 Score")
plot_bar_chart(metrics_df["train_time"], "Training Time by Model", "Seconds")
plot_bar_chart(metrics_df["test_time"], "Testing Time by Model", "Seconds")

# Section 4: Full Comparison Table
st.header("4. Full Comparison Table")

comparison_data = {
    "Model": [
        "Decision Tree (Depth=5)", "Logistic Regression", "K-Nearest Neighbors (n=5)", "LDA", "SVM",
        "Random Forest", "XGBoost", "Multilayer Perceptron", "Voting Ensemble", "Stacking"
    ],
    "Accuracy (%)": [56, 58, 67, 74, 70, 88, 91, 84, 79, 90],
    "F1 Score (%)": [48, 58, 66, 74, 70, 88, 91, 84, 79, 89],
    "Training Time (s)": [0.275, 2.981, 0.079, 0.305, 16.235, 1.933, 3.317, 268.702, 21.5, 112.941],
    "Testing Time (s)": [0.006, 0.017, 22.289, 0.0131, 0.018, 0.072, 0.082, 0.987, 55.925, 0.248],
    "Interpretability": ["Great", "Great", "Good", "Average", "Average", "Average", "Poor", "Poor", "Average", "Average"],
    "Epochs or Trees": ["NA", "100 (default)", "NA", "NA", "55", "100", "100", "55", "NA", "1000"]
}

df_comparison = pd.DataFrame(comparison_data)
st.dataframe(df_comparison, use_container_width=True)

# Optional: Display table image
st.subheader("📊 Visual Comparison Table")
try:
    image = Image.open("image.png")
    st.image(image, caption="Model Performance Comparison", use_column_width=True)
except:
    st.warning("Could not load comparison image.")

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

# Section 6: References
st.header("6. References")
st.markdown("""
- Dataset: https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data
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
