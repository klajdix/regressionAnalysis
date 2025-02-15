# **Real Estate House Price Prediction: Regression Analysis**

## **📌 Project Overview**

This project aims to predict real estate house prices using **Regression Analysis**. The dataset consists of properties from **Albania**, including details such as size, number of rooms, location, and year built. The goal is to create an accurate model that can estimate house prices based on these factors.

## **🛠️ Project Structure**

```
real_estate_price_prediction/
│
├── data/
│   ├── Albania_data_realestate.csv    # Raw dataset
│   ├── processed_data.csv             # Preprocessed dataset
│   ├── featured_data.csv              # Feature engineered dataset
│
├── models/git push origin main
│   ├── best_model.pkl                 # Trained model (Random Forest)
│   ├── preprocessor.pkl               # Preprocessing pipeline
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb   # Feature Engineering
│   ├── 0.1_data_preprocesssing.ipynb  # Data Procesing
│   ├── 04_prediction.ipynb            # Making Predictions
│
├── results/
│   ├── model_performance.csv          # Model evaluation metrics
│
├── src/
│   ├── preprocessing.py               # Data cleaning & preprocessing
│   ├── feature_engineering.py         # Feature creation
│   ├── model.py                       # Model training & evaluation
│   ├── predict.py                      # Make predictions
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
└── LICENSE                            # License file (if applicable)
```

---

## **📊 Dataset Details**

The dataset contains various features:

- **city**: The city where the property is located _(categorical)_
- **neighborhood**: Specific area within the city _(categorical)_
- **property_type**: Type of property (apartment, house, villa, etc.) _(categorical)_
- **square_meters**: Size of the property in square meters _(numerical)_
- **num_bedrooms**: Number of bedrooms _(numerical)_
- **num_bathrooms**: Number of bathrooms _(numerical)_
- **floor_level**: Floor number _(numerical)_
- **year_built**: The year the property was built _(numerical)_
- **heating_type**: Type of heating system _(categorical)_
- **distance_to_center_km**: Distance from the city center _(numerical)_
- **price_eur**: House price in EUR _(target variable, numerical)_
- **price_per_sq_meter**: Price per square meter _(calculated feature, numerical)_
- **house_age**: Age of the house _(calculated feature, numerical)_

---

## **🚀 Getting Started**

### **1️⃣ Prerequisites**

- Python 3.8 or later
- Jupyter Notebook (for exploration)
- Libraries listed in `requirements.txt`

### **2️⃣ Installation & Setup**

Clone the repository:

```bash
git clone https://github.com/klajdix/regressionAnalysis.git
cd real_estate_price_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **📌 Usage**

### **1️⃣ Data Preprocessing & Feature Engineering**

Run preprocessing and feature engineering scripts:

```bash
python src/preprocessing.py
python src/feature_engineering.py
```

**✅ Output:** `data/featured_data.csv` (processed dataset)

### **2️⃣ Train the Model**

Run the model training script:

```bash
python src/model.py
```

**✅ Outputs:**

- `models/best_model.pkl` (trained model)
- `models/preprocessor.pkl` (preprocessing pipeline)
- `results/model_performance.csv` (evaluation metrics)

### **3️⃣ Make Predictions**

Run the prediction script:

```bash
python src/predict.py
```

**✅ Output Example:**

```bash
🏡 Predicted house price: 133,141.97 EUR
```

---

## **📊 Model Performance**

| Model             | MAE (€)   | MSE (€)        | RMSE (€)  | R² Score |
| ----------------- | --------- | -------------- | --------- | -------- |
| Linear Regression | 16,439.49 | 536,582,876.09 | 23,164.25 | 0.94     |
| Ridge Regression  | 16,409.11 | 537,498,691.99 | 23,184.01 | 0.94     |
| Random Forest     | 8,577.87  | 299,746,028.79 | 17,313.17 | 0.96     |

✔ **Best Model:** Random Forest (**96% accuracy**) 🚀

---

## **💡 Future Improvements**

🔹 **Include More Features** (crime rates, nearby schools, public transport, GDP data)  
🔹 **Hyperparameter Tuning** (GridSearchCV for best model settings)  
🔹 **Deploy as an API** (Flask/FastAPI for real-time predictions)  
🔹 **Try Advanced ML Models** (XGBoost, Neural Networks, etc.)

---

## **📜 License**

This project is licensed under the MIT License.

---

###
