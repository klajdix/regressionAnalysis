import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **File Paths**
DATA_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/featured_data.csv"
MODEL_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/best_model.pkl"
PREPROCESSOR_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/preprocessor.pkl"
RESULTS_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/results/model_performance.csv"

# Ensure model directory exists
os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)

def load_data():
    """Load dataset."""
    return pd.read_csv(DATA_PATH)

def preprocess_data(df):
    """Create and fit a preprocessor for categorical and numerical features."""
    categorical_features = ["city", "neighborhood", "property_type", "heating_type"]
    numerical_features = ["square_meters", "num_bedrooms", "num_bathrooms", "floor_level", "distance_to_center_km", "house_age"]
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    
    # Drop target variable
    X = df.drop(columns=['price_eur', 'year_built'])
    y = df['price_eur']

    # Fit & transform data
    X_transformed = preprocessor.fit_transform(X)

    # ✅ Save the preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return X_transformed, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train models and evaluate performance."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2 Score": r2
        }
        
        # Select the best model based on R2 Score
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    return best_model, results

if __name__ == "__main__":
    df = load_data()
    X_transformed, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    best_model, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # ✅ Save the best model
    joblib.dump(best_model, MODEL_PATH)
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv(RESULTS_PATH)

    print(f"✅ Model training complete. Best model saved at: {MODEL_PATH}")
    print(f"🔹 Preprocessor saved at: {PREPROCESSOR_PATH}")
    print(f"📊 Model performance saved at: {RESULTS_PATH}")
