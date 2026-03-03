# Taaghche Sentiment Analysis

Refactor of the original notebook into a small Python package.

## Structure
- 	aaghche/config.py – shared configuration dataclasses (data paths, hyperparameters).
- 	aaghche/text_cleaning.py – Persian text normalization and emoji/html stripping.
- 	aaghche/data_prep.py – load CSV, trim lengths, label, clean, balance, and split.
- 	aaghche/torch_pipeline/ – PyTorch dataset, model, and training script.
- 	aaghche/tf_pipeline/ – TensorFlow dataset builders and training script.
- data/ – place 	aghche.csv here (not tracked).
- rtifacts/ – saved checkpoints (created at runtime).

## Setup
`ash
pip install -r requirements.txt
`
Key runtime deps: 	ransformers, 	orch, 	ensorflow, hazm, clean-text, 	qdm, scikit-learn, pandas.

## Running
### PyTorch
`ash
python -m taaghche.torch_pipeline.train
`
Saves best checkpoint to rtifacts/pt_model.bin.

### TensorFlow
`ash
python -m taaghche.tf_pipeline.train
`
Saves model to rtifacts/bert-fa-base-uncased-sentiment-taaghche-tf/.

Data path defaults to data/taghche.csv; update 	aaghche/config.py if your CSV lives elsewhere.
