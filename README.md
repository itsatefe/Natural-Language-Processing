# Natural-Language-Processing (Persian Sentiment)

Collection of sentiment-analysis experiments in Persian, from classical density models to modern transformers. The most up-to-date, runnable code lives in `Transformer/`.

## Repo Layout
- `Transformer/` – Refactored Taaghche review pipeline with PyTorch (`torch_pipeline/`) and TensorFlow (`tf_pipeline/`) training scripts plus shared config/cleanup utilities.
- `LSTM - Tensorflow/` – Early LSTM baseline (notebook + pretrained `sentimentAnalysis.h5`, tiny README).
- `LSTM + Attention - Tensorflow/` – LSTM with attention; includes `news_data.csv` (ignored in git) and a saved `model.keras`.
- `DensityEstimation/` – Classical ML experiments (logistic regression, Fisher’s LDA, KDE classifier, custom preprocessor) with supporting notebook, dataset, and plots.
