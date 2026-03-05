# 🏆 Model Arena — Multi-Classifier Comparison App

A full end-to-end machine learning web application that trains **12 classification models** on Apple sales data, compares their performance, and allows users to run predictions using any model of their choice.

---

## 📌 Project Overview

**Problem Statement:** Given an Apple product sales transaction, predict the `return_status` — whether the product was `Kept`, `Returned`, or `Exchanged`.

**Goal:** Train multiple ML models, compare their performance on a live dashboard, and allow users to select any model to make predictions.

---

## 🗂️ Project Structure

```
Multiple Model Comparision/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py         # Loads and splits raw data
│   │   ├── data_transformation.py    # Preprocessing pipeline
│   │   └── model_trainer.py          # Trains and evaluates all 12 models
│   ├── pipeline/
│   │   └── predict_pipeline.py       # Loads model + preprocessor for inference
│   ├── exception.py                  # Custom exception handler
│   ├── logger.py                     # Logging configuration
│   └── utils.py                      # save_object, load_object, evaluate_models
│
├── artifacts/                        # Auto-generated during training
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   ├── model_summary.json
│   └── model_<ModelName>.pkl         # One .pkl per model (12 total)
│
├── templates/
│   ├── index.html                    # Dashboard — model comparison table
│   ├── predict.html                  # Prediction form
│   └── model_info.html               # Model & metric info pages
│
├── app.py                            # Flask application
├── setup.py                          # Package setup
├── requirements.txt                  # Dependencies
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Apple Product Sales Dataset |
| **Total Rows** | 11,500 |
| **Train / Test Split** | 9,200 / 2,300 |
| **Features** | 18 |
| **Target** | `return_status` (Kept / Returned / Exchanged) |
| **Null Values** | None |
| **Class Distribution** | Kept: 88.2% · Returned: 7.9% · Exchanged: 3.9% |

### Features

| Feature | Type | Description |
|---|---|---|
| year, day | Numeric | Transaction date components |
| quarter, month, day_of_week | Categorical (Ordinal) | Time period info |
| country | Categorical (Target Encoded) | 47 countries |
| region | Categorical (OHE) | 8 global regions |
| category | Categorical (OHE) | iPhone, Mac, iPad, AirPods, Apple Watch, Accessories |
| storage | Categorical (Ordinal) | 64GB to 2TB SSD |
| unit_price_usd | Numeric | $26.69 – $7,551.01 |
| discount_pct | Numeric | 0% – 15% |
| units_sold | Numeric | 1 – 8 |
| revenue_usd | Numeric | $23 – $59,529 |
| sales_channel | Categorical (OHE) | Apple Store, Online, Carrier, B2B, Reseller |
| payment_method | Categorical (OHE) | 7 payment types |
| customer_segment | Categorical (OHE) | Business, Education, Government, Individual |
| customer_age_group | Categorical (Ordinal) | 18–24 to 55+ |
| previous_device_os | Categorical (OHE) | Android, iOS 15/16/17, New User, Other |

---

## 🤖 Models Trained

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Ridge Classifier | Linear |
| Decision Tree | Tree-Based |
| Random Forest | Ensemble |
| Gradient Boosting | Boosting |
| XGBoost | Boosting |
| CatBoost | Boosting |
| Gaussian NB | Probabilistic |
| Bernoulli NB | Probabilistic |
| KNN | Instance-Based |
| SVC | Kernel |
| LinearSVC | Linear Kernel |

---

## ⚙️ Preprocessing Pipeline

```
Numerical Features     → SimpleImputer (median) → StandardScaler
OHE Features           → SimpleImputer (most_frequent) → OneHotEncoder → StandardScaler
Target Encoded         → TargetEncoder (country)
Ordinal Features       → OrdinalEncoder
Target (return_status) → LabelEncoder → 0: Exchanged, 1: Kept, 2: Returned
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/multiple-model-comparison.git
cd multiple-model-comparison
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Run the Flask app

```bash
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

---

## 🖥️ App Features

### Dashboard (`/`)
- **Train Models** — triggers the full pipeline with a live animated loader (orbital rings + timer + model checklist)
- **Model Comparison Table** — all 12 models ranked by Test Weighted F1, with score bars and overfit detection
- **Know the Database** — modal with full dataset stats, feature breakdown, class distribution, preprocessing strategy
- **Clickable Models** — click any model name to view its definition, how it works, pros/cons, and hyperparameters
- **Clickable Metrics** — click any column header to learn about Weighted F1, Macro F1, and overfitting

### Prediction Page (`/predict_page`)
- Select any of the 12 trained models from a dropdown
- Fill in all 18 input features
- Returns prediction: **Kept**, **Returned**, or **Exchanged**

### Info Page (`/info`)
- Dynamically renders model or metric information
- Accessible via `/info?type=model&key=Random Forest`
- Accessible via `/info?type=metric&key=macro_f1`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main dashboard |
| GET | `/train` | Triggers full training pipeline |
| GET | `/models` | Returns model summary JSON |
| GET | `/predict_page` | Prediction form UI |
| POST | `/predict` | Returns prediction for given input |
| GET | `/info` | Model / metric info page |

### POST `/predict` — Example Request

```json
{
  "model_name": "Random Forest",
  "year": 2023,
  "quarter": "Q2",
  "month": "June",
  "day": 15,
  "day_of_week": 2,
  "country": "India",
  "region": "Asia",
  "category": "iPhone",
  "storage": "256 GB",
  "unit_price_usd": 999.99,
  "discount_pct": 5,
  "units_sold": 2,
  "revenue_usd": 1999.98,
  "sales_channel": "Online (Apple.com)",
  "payment_method": "Credit Card",
  "customer_segment": "Individual",
  "customer_age_group": "25–34",
  "previous_device_os": "Android"
}
```

### Response

```json
{
  "status": "success",
  "model_used": "Random Forest",
  "prediction": "Kept"
}
```

---

## 📈 Results

| Model | Test Weighted F1 | Test Macro F1 | Status |
|---|---|---|---|
| Gradient Boosting | 0.8269 | 0.3125 | ✅ Stable |
| Random Forest | 0.8269 | 0.3124 | ⚠️ Overfit |
| Bernoulli NB | 0.8269 | 0.3125 | ✅ Stable |
| LinearSVC | 0.8269 | 0.3125 | ✅ Stable |
| Decision Tree | 0.7816 | 0.3436 | ⚠️ Overfit |
| SVC | 0.7836 | 0.3331 | ✅ Stable |
| KNN | 0.5405 | 0.2612 | ⚠️ Overfit |
| Logistic Regression | 0.4491 | 0.2285 | ✅ Stable |

> **Note:** High weighted F1 with low macro F1 is expected given the class imbalance (88% Kept). The model struggles to predict minority classes (Returned, Exchanged).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn, XGBoost, CatBoost |
| Imbalance | imbalanced-learn (SMOTETomek) |
| Backend | Flask |
| Frontend | HTML, CSS, Vanilla JS (Jinja2 templates) |
| Serialization | pickle |

---

## 📦 Requirements

```
flask
scikit-learn
xgboost
catboost
imbalanced-learn
pandas
numpy
matplotlib
seaborn
```

---

## 👤 Author

**Sparsh Sharma**
sparshsharma0303@gmail.com

---

## 📄 License

This project is for educational purposes.
