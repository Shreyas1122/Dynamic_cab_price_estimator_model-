# 🚕 Dynamic Cab Pricing Predictor — Ola & Uber Fare Estimator

A machine learning project that predicts ride fares based on real-time demand and supply conditions, served through a **FastAPI** backend and consumed by a simple **HTML/JS** frontend.

---

## 📌 Overview

Ride-hailing platforms like Ola and Uber price trips dynamically based on rider demand, driver availability, trip length, and other contextual factors. This project builds a regression model that learns this pricing behavior from historical ride data and exposes it as a REST API with a working web UI on top, so a user can enter trip conditions and get an estimated fare instantly.

The project covers the full pipeline:

**Data → EDA → Preprocessing → Model Training → Evaluation → Export → API → Frontend**

---

## 🗂️ Project Structure

```
├── dynamic_pricing.ipynb     # EDA, preprocessing, model training & evaluation
├── dynamic_pricing.csv       # Raw dataset (1000 rides)
├── Dynamic_pricing.pkl       # Exported trained pipeline (preprocessing + model)
├── main.py                   # FastAPI backend serving the /predict endpoint
├── fare_estimator.html       # Frontend form that calls the API
└── README.md
```

---

## 📊 Dataset

The dataset contains **1000 historical rides** with the following original features:

| Column | Description |
|---|---|
| `Number_of_Riders` | Riders waiting/requesting rides at the time |
| `Number_of_Drivers` | Drivers available at the time |
| `Location_Category` | Urban / Suburban / Rural *(dropped — see below)* |
| `Customer_Loyalty_Status` | Regular / Silver / Gold *(dropped — see below)* |
| `Number_of_Past_Rides` | Rider's ride history count |
| `Average_Ratings` | Rider's average rating *(dropped — see below)* |
| `Time_of_Booking` | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | Economy / Premium |
| `Expected_Ride_Duration` | Trip duration in minutes |
| `Historical_Cost_of_Ride` | **Target** — the fare to predict |

No missing values or duplicate rows were found in the dataset.

---

## 🔍 Exploratory Data Analysis (EDA)

Performed using `pandas`, `matplotlib`, and `seaborn`:

- Distribution of the target variable (`Historical_Cost_of_Ride`) via histogram
- Correlation matrix + heatmap across numeric features
- Box plots and scatter plots of fare against every feature (riders, drivers, past rides, ratings, duration, location, loyalty status, time of booking, vehicle type)
- Skewness check on numeric columns — `Number_of_Drivers` showed moderate right-skew (~0.9), which informed the preprocessing choice below

**Feature selection decision:** `Location_Category`, `Customer_Loyalty_Status`, and `Average_Ratings` were dropped after EDA — `Number_of_Riders` and `Number_of_Drivers` were found to be more directly informative and already correlated with the dropped fields, so keeping all of them would add redundancy without improving the signal.

---

## ⚙️ Preprocessing Pipeline

Built with `scikit-learn` `Pipeline` and `ColumnTransformer`, split into four sub-pipelines by feature type:

| Pipeline | Applied to | Steps |
|---|---|---|
| **Skewed numeric** | `Number_of_Drivers` | `log1p` transform → `StandardScaler` |
| **Other numeric** | `Number_of_Riders`, `Number_of_Past_Rides`, `Expected_Ride_Duration` | `StandardScaler` |
| **Ordinal encoding** | `Time_of_Booking` | `OrdinalEncoder` with explicit order: Morning → Afternoon → Evening → Night |
| **One-hot encoding** | `Vehicle_Type` | `OneHotEncoder(handle_unknown="ignore")` |

All four are combined into a single `ColumnTransformer`, which is itself the first step of the final model pipeline — so preprocessing and prediction happen together as one unit.

---

## 🤖 Model

- **Algorithm:** `LinearRegression` (scikit-learn)
- **Train/test split:** 80% / 20%, `random_state=42`
- **Pipeline:** `preprocessing (ColumnTransformer) → LinearRegression`, trained end-to-end with `.fit(x_train, y_train)`

### Evaluation results (on the held-out test set)

| Metric | Score |
|---|---|
| **R² Score** | **0.875** |
| **Mean Absolute Error (MAE)** | **≈ 52.2** |

An R² of ~0.87 means the model explains about 87% of the variance in ride fares from the six input features — a strong fit for a simple linear model.

---

## 💾 Model Export

The full pipeline (preprocessing + trained model) is serialized with `joblib` into a single artifact:

```python
import joblib
joblib.dump(linear_pipeline, "Dynamic_pricing.pkl")
```

Because preprocessing is baked into the same pipeline object, the API only needs to load one `.pkl` file and can pass raw feature values straight in — no separate encoding/scaling step required at inference time.

---

## 🌐 API (FastAPI)

The trained pipeline is served through a `/predict` POST endpoint.

**Request body:**
```json
{
  "Number_of_Riders": 90,
  "Number_of_Drivers": 45,
  "Number_of_Past_Rides": 13,
  "Time_of_Booking": "Night",
  "Vehicle_Type": "Premium",
  "Expected_Ride_Duration": 90
}
```

**Response:**
```json
{
  "Estimated_price": 284.26
}
```

Field constraints are enforced server-side with Pydantic:

| Field | Type | Range |
|---|---|---|
| `Number_of_Riders` | int | 0 – 100,000 |
| `Number_of_Drivers` | int | 0 – 10,000 |
| `Number_of_Past_Rides` | int | 0 – 100,000 |
| `Time_of_Booking` | enum | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | enum | Premium / Economy |
| `Expected_Ride_Duration` | int | 0 – 300 |

### Run the API

```bash
pip install fastapi uvicorn scikit-learn joblib pandas
uvicorn main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`.

---

## 🖥️ Frontend

`fare_estimator.html` is a self-contained dark-themed web form ("Trip details") that:

- Collects riders waiting, drivers available, rider's past rides, duration, time of booking, and vehicle type
- Validates every field client-side against the same limits as the backend before sending anything
- Calls `POST http://127.0.0.1:8000/predict`
- Displays the estimated fare, or a readable error message if the API rejects the request or is unreachable

Just open the HTML file in a browser while the FastAPI server is running.

---

## 🛠️ Tech Stack

- **Data & ML:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** HTML, CSS, vanilla JavaScript

---


