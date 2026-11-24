# Dry Bean Classification — Streamlit App

A Streamlit application for Dry Bean classification using four machine learning algorithms:
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (GaussianNB)

## Run locally

1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

5. Upload the original Dry Bean dataset as CSV or Excel. The dataset must contain a target column named `Class`.

## Deploy using GitHub and Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Open https://share.streamlit.io/ and sign in using GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Deploy.

The dataset is uploaded by the user inside the app, so you do not need to commit the dataset to GitHub.

## Notes

- The app uses an 80/20 stratified train/test split by default.
- StandardScaler is applied inside each model pipeline.
- Results and trained model files can be downloaded from the app.
- The app reports test-set performance; results may vary if split settings or dataset contents change.
