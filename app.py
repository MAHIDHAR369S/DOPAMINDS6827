import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

st.set_page_config(page_title="Parkinson's Early Detection", layout="wide")

# Load dataset with proper encoding to avoid UTF-8 error
df = pd.read_csv("parkinsons.data", encoding='latin1')
df.drop(columns=["name"], inplace=True)

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

st.title("🧠 Parkinson’s Disease Early Detection using XGBoost")

st.subheader("Model Evaluation")
col1, col2 = st.columns(2)

with col1:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(8, 6))
importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:10]
ax.barh(X.columns[indices], importance[indices])
ax.invert_yaxis()
st.pyplot(fig)

st.subheader("Predict Parkinson’s Disease")
input_data = []

for feature in X.columns:
    value = st.number_input(feature, value=0.0)
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Parkinson’s Disease Detected (Probability: {probability:.2f})")
    else:
        st.success(f"Healthy Individual (Probability: {1 - probability:.2f})")
