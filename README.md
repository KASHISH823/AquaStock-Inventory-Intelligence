# 🐠 AquaStock – Inventory Intelligence System

### Machine Learning-Based Demand Forecasting & Smart Inventory Management

<p align="center">
  <b>Turning demand predictions into smarter inventory decisions.</b>
</p>

---

## 📌 About the Project

**AquaStock** is a machine learning-based inventory intelligence system designed for aquatic and aquarium-related products.

Managing inventory is a challenging task for businesses. Maintaining too little stock can lead to product shortages, while keeping excessive inventory can block working capital and increase storage costs.

AquaStock addresses this problem by combining **Machine Learning-based Demand Forecasting** with practical inventory management rules.

The system predicts monthly product demand using historical data and uses these predictions to calculate safety stock, target stock, stock coverage and reorder quantities.

The results are presented through an interactive **Streamlit web application**, helping users understand inventory conditions and make more informed stocking decisions.

## 🎯 Project Objectives

- Forecast monthly demand for aquatic products using machine learning.
- Analyse historical demand patterns and product-related information.
- Compare predicted demand with available inventory.
- Calculate safety stock, target stock and reorder quantities.
- Identify potentially overstocked and slow-moving products.
- Provide interactive inventory insights through a Streamlit dashboard.
- Allow users to explore demand-change scenarios using a What-If Simulator.

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Executive Dashboard | Provides an overall view of demand and inventory indicators. |
| 📈 Demand Forecasting | Displays predicted monthly demand and demand trends. |
| 📦 Inventory Health | Identifies stock conditions using stock coverage. |
| 🔄 Smart Reorder | Calculates suggested reorder quantities. |
| 🐢 Slow-Moving Products | Highlights products with relatively low predicted demand. |
| 🔍 Product Intelligence | Provides product-level inventory insights. |
| 🎛️ What-If Simulator | Explores how demand changes affect inventory requirements. |
| 🤖 Model Performance | Compares machine learning models using evaluation metrics. |

## 🧠 Machine Learning Approach

AquaStock uses a **supervised machine learning regression approach** to estimate monthly product demand.

The target variable represents the numerical number of orders expected for a product line in a city.

### Models Evaluated

The project compares baseline forecasting techniques with machine learning algorithms:

- Naive – Last Month
- Naive – 3-Month Average
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Tuned XGBoost Regressor

### Model Evaluation

The models are evaluated using three regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted demand.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalising larger errors more heavily.
- **R² Score:** Measures how much variation in the target values is explained by the model.

### Model Comparison Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Naive – Last Month | 8.57 | 10.96 | 0.707 |
| Naive – 3-Month Average | 6.82 | 8.74 | 0.814 |
| Linear Regression | 5.58 | 7.09 | 0.877 |
| Random Forest | 5.54 | 7.05 | 0.879 |
| XGBoost | 5.60 | 7.13 | 0.876 |
| Tuned XGBoost | **5.51** | **7.01** | **0.880** |

**Final Model Selected: Random Forest Regressor**

Random Forest was selected for the working prototype because its performance was almost identical to tuned XGBoost while providing a practical and relatively straightforward model to work with.

*Note: Tuned XGBoost achieved marginally better recorded evaluation metrics.*

## ⚙️ Inventory Decision Logic

AquaStock goes beyond demand prediction by converting forecast values into practical inventory indicators.

The system uses the following calculations:

**1. Stock Gap**

Current Stock − Predicted Demand

**2. Safety Stock**

Predicted Demand × 10%

**3. Target Stock**

Predicted Demand + Safety Stock

**4. Reorder Quantity**

max(Target Stock − Current Stock, 0)

**5. Excess Stock**

max(Current Stock − Target Stock, 0)

**6. Stock Coverage**

Current Stock ÷ Predicted Demand

The prototype uses a 10% safety-stock buffer as a project assumption.

Stock coverage is used to classify inventory conditions:

- Less than 1 month: Stockout Risk
- 1–2 months: Healthy
- More than 2 months: Overstocked

These thresholds are prototype rules and should be adjusted using actual supplier lead times, demand variability and business requirements in a real-world implementation.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and model evaluation |
| Random Forest | Selected demand forecasting model |
| XGBoost | Model comparison and hyperparameter tuning |
| Matplotlib | Data visualisation |
| Seaborn | Exploratory data analysis |
| Streamlit | Interactive web application |
| Joblib | Saving and loading the trained model |
| Jupyter Notebook | EDA, modelling and experimentation |
| VS Code | Development environment |

## 📂 Project Structure

```text
aquatic-inventory-ml/
│
├── app.py
├── model.pkl
├── feature_list.json
├── prepare_dataset.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── aquatic_demand_ml_dataset.csv
│   ├── aquatic_transactions_clean_10k.csv
│   ├── inventory_business_output.csv
│   ├── model_results.csv
│   └── test_predictions.csv
│
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_modeling.ipynb
    └── 03_business_logic.ipynb
```

### File Description

- `app.py` – Main Streamlit dashboard application.
- `model.pkl` – Saved trained machine learning model.
- `feature_list.json` – Stores the model feature information.
- `prepare_dataset.py` – Dataset preparation script.
- `requirements.txt` – Required Python dependencies.
- `data/` – Prepared datasets, model results and inventory output files.
- `notebooks/` – Exploratory data analysis, model development and inventory business logic.

## 🚀 Installation & Setup

Follow these steps to run AquaStock locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AquaStock-Inventory-Intelligence.git
```

### 2. Navigate to the Project Directory

```bash
cd AquaStock-Inventory-Intelligence
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will normally open in your browser at:

`http://localhost:8501`

## 📊 System Workflow

```text
Historical Dataset
       ↓
Data Preparation & EDA
       ↓
Feature Engineering
       ↓
Model Training & Comparison
       ↓
Random Forest Model Selection
       ↓
Monthly Demand Prediction
       ↓
Inventory Business Logic
       ↓
Safety Stock & Target Stock
       ↓
Reorder & Stock Coverage
       ↓
Streamlit Dashboard
       ↓
Inventory Decision Support
```

## ⚠️ Limitations

- The model is evaluated on a limited later time period, so additional historical observations are needed for stronger validation.
- The 10% safety-stock buffer is an assumption used in this academic prototype.
- Supplier lead time, minimum order quantity, purchase costs and shelf-life constraints are not fully incorporated.
- The application uses saved model outputs and prepared inventory data rather than retraining the model whenever the dashboard opens.
- The What-If Simulator explores business-rule scenarios; it does not retrain the forecasting model.
- Forecasts and inventory recommendations are decision-support estimates, not guaranteed sales or profit outcomes.

## 🔮 Future Enhancements

- Integrate real-time inventory and sales data.
- Incorporate supplier lead times and minimum order quantities.
- Improve forecasting with additional historical data.
- Introduce automated model retraining and monitoring.
- Develop more advanced inventory optimisation techniques.
- Integrate database connectivity for live inventory updates.

## 💡 Project Highlights

The main contribution of AquaStock is the integration of **Machine Learning-based Demand Forecasting and Inventory Decision Support** in a single application.

Instead of simply predicting future demand, the system translates those predictions into understandable inventory indicators and suggested actions.

This makes the project a practical demonstration of how data analytics and machine learning can support business decision-making.

## 👩‍💻 Author

**Kashish**

BCA – Honors and Research

### ⭐ If you find this project interesting, consider giving this repository a star!

