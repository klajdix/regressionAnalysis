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
├── models/
│   ├── best_model.pkl                 # Trained model (Random Forest)
│   ├── preprocessor.pkl               # Preprocessing pipeline
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Data preprocessing & EDA
│   ├── 02_model_building.ipynb        # Model training & evaluation
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
- **city**: The city where the property is located *(categorical)*
- **neighborhood**: Specific area within the city *(categorical)*
- **property_type**: Type of property (apartment, house, villa, etc.) *(categorical)*
- **square_meters**: Size of the property in square meters *(numerical)*
- **num_bedrooms**: Number of bedrooms *(numerical)*
- **num_bathrooms**: Number of bathrooms *(numerical)*
- **floor_level**: Floor number *(numerical)*
- **year_built**: The year the property was built *(numerical)*
- **heating_type**: Type of heating system *(categorical)*
- **distance_to_center_km**: Distance from the city center *(numerical)*
- **price_eur**: House price in EUR *(target variable, numerical)*
- **price_per_sq_meter**: Price per square meter *(calculated feature, numerical)*
- **house_age**: Age of the house *(calculated feature, numerical)*

---
## **🚀 Getting Started**
### **1️⃣ Prerequisites**
- Python 3.8 or later
- Jupyter Notebook (for exploration)
- Libraries listed in `requirements.txt`

### **2️⃣ Installation & Setup**
Clone the repository:
```bash
git clone https://github.com/yourusername/real_estate_price_prediction.git
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
| Model               | MAE (€)  | MSE (€)         | RMSE (€)       | R² Score |
|---------------------|---------|----------------|---------------|----------|
| Linear Regression  | 16,439.49 | 536,582,876.09 | 23,164.25     | 0.94     |
| Ridge Regression   | 16,409.11 | 537,498,691.99 | 23,184.01     | 0.94     |
| Random Forest      | 8,577.87  | 299,746,028.79 | 17,313.17     | 0.96     |

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
### **🚀 Ready to Deploy? Let me know if you need help!** 🎯
