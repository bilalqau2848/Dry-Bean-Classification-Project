import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

st.title("🫘 Dry Bean Classification")
st.write(
    "Machine learning application for classifying dry bean varieties "
    "using LDA, QDA, KNN and Gaussian Naive Bayes."
)

st.sidebar.header("Project Information")
st.sidebar.write("Dataset: UCI Dry Bean Dataset")
st.sidebar.write("Models: LDA, QDA, KNN, GaussianNB")
st.sidebar.caption("Upload the original dataset to begin.")

uploaded_file = st.file_uploader(
    "Upload the original Dry Bean dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

def read_dataset(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def make_models():
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB()
    }

if uploaded_file is not None:
    try:
        df = read_dataset(uploaded_file)
        df.columns = df.columns.str.strip()

        if "Class" not in df.columns:
            st.error("The uploaded file must contain a target column named 'Class'.")
            st.stop()

        df = df.dropna().copy()
        X = df.drop(columns=["Class"]).apply(pd.to_numeric, errors="coerce")
        valid = X.notna().all(axis=1)
        X = X.loc[valid]
        y = df.loc[valid, "Class"].astype(str)

        if X.empty:
            st.error("No usable numeric feature rows were found.")
            st.stop()

        st.success("Dataset loaded successfully.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", f"{len(df):,}")
        c2.metric("Features", X.shape[1])
        c3.metric("Classes", y.nunique())

        with st.expander("Preview dataset"):
            st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Class Distribution")
        counts = y.value_counts().rename_axis("Class").reset_index(name="Count")
        st.bar_chart(counts.set_index("Class"))

        test_size = st.sidebar.slider(
            "Test set size (%)", min_value=10, max_value=40, value=20, step=5
        )
        random_state = st.sidebar.number_input(
            "Random state", min_value=0, value=42, step=1
        )

        if st.button("🚀 Train and Evaluate Models", type="primary"):
            if y.value_counts().min() < 2:
                st.error("Each class needs at least 2 records for stratified splitting.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size / 100,
                random_state=int(random_state),
                stratify=y
            )

            models = make_models()
            trained_models = {}
            result_rows = []
            predictions = {}

            with st.spinner("Training and evaluating models..."):
                for name, classifier in models.items():
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("classifier", classifier)
                    ])
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)

                    trained_models[name] = pipe
                    predictions[name] = pred
                    result_rows.append({
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, pred),
                        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                        "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0)
                    })

            results = pd.DataFrame(result_rows).sort_values("Accuracy", ascending=False)
            st.session_state["trained_models"] = trained_models
            st.session_state["feature_names"] = list(X.columns)
            st.session_state["class_names"] = sorted(y.unique().tolist())
            st.session_state["results"] = results
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["predictions"] = predictions
            st.session_state["dataset_columns"] = list(X.columns)

        if "results" in st.session_state:
            st.subheader("Model Performance")
            st.dataframe(
                st.session_state["results"].style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}"
                }),
                use_container_width=True
            )

            best = st.session_state["results"].iloc[0]
            st.success(
                f"Best model by test accuracy: {best['Model']} "
                f"({best['Accuracy'] * 100:.2f}%)"
            )

            st.subheader("Accuracy Comparison")
            chart_data = st.session_state["results"].set_index("Model")[["Accuracy"]]
            st.bar_chart(chart_data)

            st.subheader("Confusion Matrices")
            for name, model in st.session_state["trained_models"].items():
                pred = st.session_state["predictions"][name]
                labels = model.classes_
                cm = confusion_matrix(st.session_state["y_test"], pred, labels=labels)

                with st.expander(f"{name} — Confusion Matrix and Report"):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels, ax=ax
                    )
                    ax.set_xlabel("Predicted class")
                    ax.set_ylabel("Actual class")
                    ax.set_title(f"Confusion Matrix: {name}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    report = classification_report(
                        st.session_state["y_test"],
                        pred,
                        zero_division=0
                    )
                    st.code(report)

            st.subheader("5-Fold Cross Validation")
            if st.button("Run 5-Fold Cross Validation"):
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_rows = []
                for name, classifier in make_models().items():
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("classifier", classifier)
                    ])
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
                    cv_rows.append({
                        "Model": name,
                        "Mean CV Accuracy": scores.mean(),
                        "Standard Deviation": scores.std()
                    })
                cv_df = pd.DataFrame(cv_rows).sort_values(
                    "Mean CV Accuracy", ascending=False
                )
                st.dataframe(
                    cv_df.style.format({
                        "Mean CV Accuracy": "{:.4f}",
                        "Standard Deviation": "{:.4f}"
                    }),
                    use_container_width=True
                )
                st.download_button(
                    "Download Cross-Validation Results CSV",
                    cv_df.to_csv(index=False).encode("utf-8"),
                    "Cross_Validation_Results.csv",
                    "text/csv"
                )

            st.download_button(
                "Download Model Comparison CSV",
                st.session_state["results"].to_csv(index=False).encode("utf-8"),
                "Dry_Bean_Classification_Results.csv",
                "text/csv"
            )

            st.subheader("Download Trained Models")
            for name, model in st.session_state["trained_models"].items():
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                st.download_button(
                    f"Download {name} model",
                    data=buffer.getvalue(),
                    file_name=f"{name}_Dry_Bean_Model.pkl",
                    mime="application/octet-stream",
                    key=f"download_{name}"
                )

            st.subheader("Try a Prediction")
            st.write("Enter measurements for one dry bean sample.")
            selected_model_name = st.selectbox(
                "Choose a trained model",
                list(st.session_state["trained_models"].keys())
            )
            selected_model = st.session_state["trained_models"][selected_model_name]

            feature_cols = st.session_state["feature_names"]
            sample_values = {}
            cols = st.columns(3)
            for i, feature in enumerate(feature_cols):
                series = X[feature]
                default_value = float(series.median())
                min_value = float(series.min())
                max_value = float(series.max())
                if min_value == max_value:
                    sample_values[feature] = default_value
                else:
                    with cols[i % 3]:
                        sample_values[feature] = st.number_input(
                            feature,
                            min_value=min_value,
                            max_value=max_value,
                            value=default_value,
                            key=f"feature_{feature}"
                        )

            if st.button("Predict Bean Class"):
                sample_df = pd.DataFrame([sample_values], columns=feature_cols)
                prediction = selected_model.predict(sample_df)[0]
                st.success(f"Predicted Dry Bean Class: **{prediction}**")

    except Exception as e:
        st.error(f"Could not process the dataset: {e}")
        st.info("Check that you uploaded the original Dry Bean dataset with a 'Class' column.")

else:
    st.info("Please upload the original Dry Bean dataset to train models and use the prediction feature.")
    st.markdown(
        """
        **Expected dataset columns:** 16 numeric measurement features and one target column named `Class`.

        The application supports CSV and Excel files.
        """
    )

st.divider()
st.caption("Dry Bean Classification Project | Statistical Pattern Recognition")
