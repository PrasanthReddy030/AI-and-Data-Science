# AI and Data Science Portfolio

A collection of machine learning and deep learning projects.

## Structure

```
AI-and-Data-Science/
├── Machine-Learning-Projects/
│   ├── House Price Predictor
│   └── Titanic Survival Predictor
└── Deep-Learning-Projects/
    ├── MNIST Digit Classifier
    └── distilbert_news_classifier
```

## Machine Learning Projects

| Project | Description | Key Techniques |
|---|---|---|
| [House Price Predictor](Machine-Learning-Projects/House%20Price%20Predictor) | Regression on the Kaggle House Prices dataset — predicting Ames, Iowa home sale prices from 80 features | Ridge/Lasso regression, GridSearchCV, SHAP explainability |
| [Titanic Survival Predictor](Machine-Learning-Projects/Titanic%20Survival%20Predictor) | Binary classification on Titanic passenger data | Logistic Regression, Decision Tree, SVM, KNN, sklearn Pipeline |

## Deep Learning Projects

| Project | Description | Key Techniques |
|---|---|---|
| [MNIST Digit Classifier](Deep-Learning-Projects/MNIST%20Digit%20Classifier) | Handwritten digit classification, built from scratch in pure PyTorch (no Trainer API) | Feedforward neural network, manual training loop, Adam |
| [DistilBERT News Classifier](Deep-Learning-Projects/distilbert_news_classifier) | Fine-tuned DistilBERT on AG News, benchmarked against a TF-IDF + Logistic Regression baseline | HuggingFace Trainer, transfer learning, early stopping, W&B experiment tracking |
