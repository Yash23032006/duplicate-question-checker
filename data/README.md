# Training dataset

`train.csv` is intentionally not included in the deployment bundle. The Streamlit app does not need it because it loads the pre-trained model from `../models/duplicate_model_char.joblib`.

To retrain or evaluate locally, obtain the dataset from its original source, check its licence/redistribution terms, and place the CSV here as `data/train.csv`. Expected columns: `question1`, `question2`, and `is_duplicate`.
