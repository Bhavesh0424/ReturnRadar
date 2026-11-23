# ReturnRadar — Product Return Prediction

ReturnRadar is a beginner-friendly machine learning portfolio project that estimates the likelihood that an e-commerce order may be returned. It demonstrates a reproducible ML workflow: data generation, preprocessing, model training, evaluation, and prediction for a new order.

## Important data and model note

**This repository uses synthetic demonstration data, not real customer transactions.** The generated patterns are artificial. Evaluation metrics from this dataset do not establish real-world accuracy or business performance. Before using a model operationally, replace the demo data with appropriately collected real order data, validate it on a representative holdout set, and review fairness, privacy, and business risks.

The prediction is decision support only. It must not be used by itself to deny an order or a customer's return/refund rights.

## Features

- Generates a reproducible synthetic order dataset.
- Preprocesses numerical and categorical features in a scikit-learn pipeline.
- Trains a Random Forest classifier.
- Prints precision, recall, F1-score, and ROC-AUC.
- Prompts for one order and displays estimated return probability and a Low / Medium / High demo risk band.

## Project structure

```text
ReturnRadar/
├── data/
│   └── (synthetic_orders.csv is generated when training)
├── models/
│   └── (return_model.joblib is generated when training)
├── src/
│   ├── data_generator.py
│   ├── train_model.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- Internet connection for installing Python packages

## Run on Windows

Open Command Prompt in the project folder and run:

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\train_model.py
.venv\Scripts\python.exe src\predict.py
```

The first training run creates `data/synthetic_orders.csv` and `models/return_model.joblib`. The model file is intentionally excluded from Git by `.gitignore`; recruiters can reproduce it by running the training command.

## Example input

Try these values when prompted:

- Age: `28`
- Previous purchases: `5`
- Historical return rate: `0.20`
- Category: `Clothing`
- Price: `499`
- Discount: `10`
- Rating: `4.2`
- Payment: `UPI`
- Product views: `4`
- Session duration: `8`
- Shipping: `Standard`
- Coupon: `1`

The output includes an estimated probability and a demo risk band. Your result may differ if you change the inputs.

## Model evaluation

The training script prints a classification report and ROC-AUC on a held-out test split. Because the dataset is synthetic, these metrics are only a software demonstration and should not be interpreted as real-world performance.

## Future improvements

- Train and validate using a real, appropriately licensed e-commerce returns dataset.
- Compare Logistic Regression, Random Forest, and Gradient Boosting.
- Add cross-validation, calibration, and feature-importance analysis.
- Document data limitations, fairness checks, and model monitoring.

## Author

Created as a machine learning portfolio project.
