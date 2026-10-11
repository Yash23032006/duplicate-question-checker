> **Deployment note (GitHub + Streamlit Community Cloud):** The deployed app needs `app.py`, `preprocessing.py`, `requirements.txt`, and `models/duplicate_model_char.joblib`. The training dataset is intentionally not included in this deployment bundle because it is large and is not needed for predictions. Check the dataset licence before publishing it separately. Before deployment, pin `scikit-learn` in `requirements.txt` to the exact version used when the model file was trained (instructions below); scikit-learn model files are not guaranteed to work across versions.

# Duplicate Question Checker (NLP Mini Project)

## 1. Project Overview

The **Duplicate Question Checker** is a Natural Language Processing (NLP) mini project that identifies whether two questions express essentially the same meaning.

The application uses **TF-IDF vectorization, word-level and character-level features, and a Linear Support Vector Machine (SVM)** to classify question pairs as duplicate or non-duplicate. A Streamlit interface allows users to enter two questions and view the prediction and analysis results.

This project demonstrates how traditional machine learning and NLP techniques can be applied to question similarity detection.

## 2. Problem Statement

Online question-and-answer platforms often contain multiple questions asking for the same information. Identifying duplicate questions can help reduce repetition and improve information organization.

The objective of this project is to build an NLP application that compares two questions and predicts whether they are duplicates.

## 3. Project Objectives

- Accept two questions as input.
- Preprocess the question text.
- Convert text into numerical features using TF-IDF.
- Combine word-level and character-level features.
- Predict whether the questions are duplicates.
- Display the prediction and relevant analysis results.
- Evaluate the model using standard machine learning metrics.

## 4. Features

- User-friendly Streamlit web interface.
- Duplicate and non-duplicate question classification.
- Word-level and character-level TF-IDF features.
- Linear SVM classification.
- Word-vector cosine similarity display.
- SVM decision score display.
- Model training and evaluation scripts.
- Confusion matrices and misclassification analysis.
- A separate baseline model for comparison.

## 5. Technologies Used

- **Programming language:** Python
- **NLP technique:** Text preprocessing and TF-IDF vectorization
- **Feature extraction:** Word n-grams and character n-grams
- **Machine learning algorithm:** Linear Support Vector Machine (SVM)
- **Baseline experiments:** Logistic Regression and Linear SVM
- **Web framework:** Streamlit
- **Data processing:** Pandas and NumPy
- **Machine learning tools:** Scikit-learn
- **Feature matrix processing:** SciPy
- **Model persistence:** Joblib
- **Evaluation and visualization:** Scikit-learn, Matplotlib, and Seaborn

## 6. Dataset

The project uses a question-pair dataset stored at:

`data/train.csv`

The dataset contains pairs of questions and labels indicating whether each pair is considered duplicate or non-duplicate.

The training scripts use a reproducible sample of up to 50,000 rows, with a stratified training/test split.

The dataset is used for training and evaluation. The trained model bundle allows the application to make predictions without loading the training dataset at runtime.

**Note:** Check the dataset's license and redistribution requirements before sharing it publicly.

## 7. Methodology

### Step 1: Text Preprocessing

Question text is converted to lowercase, repeated whitespace is normalized, and missing text is handled safely.

### Step 2: TF-IDF Feature Extraction

The model uses two types of TF-IDF features:

- **Word-level TF-IDF:** Captures individual words and word sequences.
- **Character-level TF-IDF:** Captures character patterns that can help with spelling variations and related word forms.

### Step 3: Pairwise Feature Construction

For each question pair, the model constructs features using the absolute difference and element-wise product of the two question vectors.

The word-level and character-level pair features are then combined into a single feature representation.

### Step 4: Classification

A Linear SVM uses the combined feature representation to predict whether the question pair is duplicate or non-duplicate.

### Step 5: Prediction and Analysis

The Streamlit application displays the predicted class, word-vector cosine similarity, and SVM decision score.

The cosine similarity and SVM decision score provide different information. Neither should be interpreted directly as the probability that two questions are duplicates.

## 8. Project Structure

```text
duplicate_question_checker/
├── app.py
├── app_word_baseline.py
├── preprocessing.py
├── train_model.py
├── train_model_char.py
├── train_model_embeddings.py
├── evaluate_model.py
├── analyze_errors.py
├── debug_prediction.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── train.csv
├── models/
│   ├── duplicate_model.joblib
│   └── duplicate_model_char.joblib
├── evaluation_results/
│   ├── confusion_matrix_linear_svm.png
│   ├── confusion_matrix_logistic_regression.png
│   └── misclassified_linear_svm.csv
└── screenshots/
```

The `screenshots/` directory is optional and should contain genuine screenshots of the running application if you choose to include it.

The `venv/` and `__pycache__/` directories are omitted from this listing because they should not be included in the final submission ZIP.

## 9. Requirements

- Python 3.10 or newer recommended.
- pip package installer.
- The project dependencies listed in `requirements.txt`.
- The trained model bundle `models/duplicate_model_char.joblib`.
- The dataset at `data/train.csv` for training and evaluation.

## 10. Installation and Setup

### Windows

Open a terminal in the project directory.

If you already have a working virtual environment, activate it:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

If setting up the project on a new computer, create a virtual environment first:

```powershell
py -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

### macOS/Linux

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## 11. Training the Model

To train the word-plus-character model, run the following command from the project root:

```bash
python train_model_char.py
```

The script trains the model using word-level and character-level TF-IDF features and saves the trained model bundle to:

`models/duplicate_model_char.joblib`

**Important:** The training script may overwrite the existing character-model bundle. Since the current application already works, retraining is unnecessary unless you intend to repeat the experiment.

The original baseline model can be trained separately using:

```bash
python train_model.py
```

Follow the relevant training script's output and configuration to determine which model file it creates.

## 12. Evaluating the Model

Run the evaluation script:

```bash
python evaluate_model.py
```

The script evaluates the models supported by its current implementation and reports metrics and confusion matrices.

Evaluation output may be saved in:

`evaluation_results/`

Available evaluation artifacts in this project include confusion matrix images and a CSV of misclassified Linear SVM examples.

Use results from the matching training and evaluation run when reporting performance.

## 13. Running the Application

Make sure the trained model bundle exists and the required dependencies are installed.

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually:

`http://localhost:8501`

Open the URL in your browser to use the application.

The current application should load:

`models/duplicate_model_char.joblib`

The model bundle must contain the model and the matching word and character vectorizers expected by `app.py`.

The training dataset is not required for ordinary predictions after the trained bundle has been created.

## 14. Example Test Cases

| Question 1 | Question 2 | Expected result |
|---|---|---|
| How can I learn Python? | What is the best way to learn Python? | Duplicate |
| What is machine learning? | Explain machine learning. | Duplicate |
| How do I cook rice? | What is the capital of India? | Non-duplicate |
| How do I prepare for the GRE? | How should I plan my preparation for GRE? | Duplicate |

These are illustrative manual test cases, not a guarantee of the model's output. The model may misclassify some semantically similar questions.

In manual testing, the application correctly classified the Python question pair, the machine-learning question pair, and the unrelated cooking/capital question pair. The GRE preparation pair was misclassified as non-duplicate.

These observations demonstrate both successful predictions and a limitation of the current model.

## 15. Evaluation Results

In one evaluation run, the Word + Character TF-IDF Linear SVM achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 78.85% |
| Precision | 71.13% |
| Recall | 72.30% |
| F1-score | 71.71% |
| ROC-AUC | 86.35% |

These results correspond to a particular dataset sample and train/test split. They should be treated as measured experimental results, not guaranteed performance on unseen data.

Accuracy measures the proportion of correct predictions. Precision measures how often positive duplicate predictions are correct. Recall measures how many actual duplicate pairs are identified. F1-score balances precision and recall, while ROC-AUC measures how well the model ranks positive examples above negative examples across decision thresholds.

The model can still make incorrect predictions, particularly when questions express similar meanings using different wording.

## 16. Limitations

- TF-IDF primarily represents word and character patterns rather than deep semantic meaning.
- Questions with different wording but similar intent may be classified incorrectly.
- Word-vector cosine similarity is not a duplicate probability.
- Performance depends on dataset quality, sampling, and the train/test split.
- Evaluation results on a random pair split may be optimistic if related questions occur across both training and test data.
- The current application performs binary classification; it does not generate answers to questions.

## 17. Future Improvements

Possible future improvements include:

- Evaluating sentence embeddings to capture semantic similarity more effectively.
- Comparing traditional TF-IDF models with transformer-based approaches.
- Improving the quality and diversity of the training data.
- Investigating false positives and false negatives.
- Evaluating on a separate held-out test set and checking for question overlap between splits.
- Improving the interface and adding confidence calibration if appropriate.

Any improved model should be compared against the current baseline using a consistent evaluation procedure.

## 18. GitHub and Submission Checklist

Before sharing or submitting the project:

1. Verify that `app.py` loads `models/duplicate_model_char.joblib`.
2. Include the corresponding trained model bundle if the application must run without retraining.
3. Include `requirements.txt`, the source files, and this README.
4. Exclude `venv/`, `__pycache__/`, and other unnecessary generated files from the submission ZIP.
5. Check the dataset's license, size, and redistribution requirements before publishing it.
6. Check the trained model file size and repository hosting limits before committing it to GitHub.
7. Include genuine application screenshots if required by your college.
8. Test the application after preparing the final submission.

## 19. Conclusion

The Duplicate Question Checker demonstrates how Natural Language Processing and traditional machine learning can be used to identify potentially duplicate questions.

By combining word-level and character-level TF-IDF features with a Linear SVM classifier, the project provides a working question-pair classification application with a browser-based interface and evaluation tools.

The measured results show that the model can identify many duplicate and non-duplicate question pairs, while its mistakes highlight the challenges of semantic similarity detection and opportunities for future improvement.

## 20. GitHub and Streamlit Community Cloud Deployment

### A. Final checks before pushing

1. Confirm `models/duplicate_model_char.joblib` exists. `app.py` uses this word + character TF-IDF model.
2. Confirm `app.py`, `preprocessing.py`, `requirements.txt`, and the model file are committed.
3. Do not commit virtual environments, secrets, or generated evaluation files.
4. The original `data/train.csv` dataset is excluded by `.gitignore`. It is only needed if you want to retrain or rerun the evaluation scripts; the deployed app does not need it.
5. **Important model compatibility check:** run `python -c "import sklearn; print(sklearn.__version__)"` in the same Python environment in which this model was trained. Replace the `scikit-learn` line in `requirements.txt` with `scikit-learn==THE_VERSION_PRINTED`. Do not literally use `THE_VERSION_PRINTED`. If you no longer have the training environment, test the deployment carefully; the version cannot be reliably inferred from the `.joblib` file name.
6. Run the app locally before pushing:

   ```bash
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

7. Test both empty inputs and normal question pairs, and confirm the model loads without an error.

### B. Push to GitHub

Create an empty repository on GitHub. From the project root, run:

```bash
git init
git add .
git status
```

Review `git status` and ensure `data/train.csv`, `.env`, virtual environments, and unwanted files are not staged. Then run:

```bash
git commit -m "Prepare Duplicate Question Checker for deployment"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your actual GitHub repository URL. If this folder is already a Git repository, do not run `git init` again; inspect `git status` and `git remote -v` first.

### C. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app** (or **Deploy an app**).
3. Choose the GitHub repository and the `main` branch.
4. Set the main file path to `app.py` (the file is in the repository root).
5. Deploy the app and wait for dependency installation and startup.
6. Open the generated public app URL and test at least one duplicate pair and one non-duplicate pair.

### D. If deployment fails

- **Model loading / scikit-learn error:** ensure the `scikit-learn` version in `requirements.txt` matches the training environment, then commit and push the change.
- **Missing model file:** check that `models/duplicate_model_char.joblib` was committed and that the path/capitalization is exactly correct.
- **Dependency installation error:** read the first package error in the Streamlit app logs, adjust `requirements.txt`, then commit and push.
- **GitHub rejects a file:** check `git status` and the file size. The dataset is excluded in this bundle; avoid committing large training data unless you have verified redistribution rights and a good reason to host it.
- **App works locally but not in the cloud:** inspect the Streamlit app logs; local and cloud Python/package versions may differ.

After changing files, push the commit to GitHub. Streamlit Community Cloud normally detects the repository update and rebuilds the app.
