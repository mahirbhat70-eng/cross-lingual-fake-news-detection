# Cross Lingual Fake News Detection

This project implements a cross lingual fake news detection pipeline using XLM RoBERTa based models. Code includes preprocessing, training, evaluation, and a minimal app for inference.

## Quickstart
1. Create virtual env and activate
   .\.venv\Scripts\Activate.ps1
2. Install dependencies
   pip install -r requirements.txt
3. Train
   python src/train.py
4. Run app
   python scripts/app.py

## Run Prediction
```bash
python src/predict.py --text "Example news article"
```

## Trained Model

Due to GitHub file size limits, trained models are not stored in this repository.

To use the model:

1. Download the trained checkpoint from the link below
   `[MODEL_DOWNLOAD_LINK]`
2. Place the downloaded model inside
   `models/xlm_roberta_fake_news_classifier/`
3. Run inference
   `python src/predict.py`

## Dataset

This project uses multiple datasets for cross lingual fake news detection.

Example datasets:
- LIAR dataset
- Hindi news dataset
- Politifact dataset

Datasets are not included in this repository. Place datasets inside:
`data/dataset/`

## Project Pipeline

1. **Data Collection**
   News datasets collected from multiple sources.
2. **Data Preprocessing**
   Cleaning text and preparing multilingual inputs.
3. **Model Training**
   Fine tuning XLM RoBERTa for fake news classification.
4. **Evaluation**
   Model evaluated on validation and test sets.
5. **Deployment**
   Inference through the app.py interface.

## Results

Model: XLM RoBERTa

- Accuracy: XX %
- F1 Score: XX %

Results and evaluation metrics are stored in the results/ directory.

## Files
- src : training and model code
- scripts : app and helper scripts
- notebooks : EDA and experiments
- data : raw and processed data (ignored)
- models : trained checkpoints (ignored)
