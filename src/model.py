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

INPUT_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/featured_data.csv"
MODEL_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/best_model.pkl"
PREPROCESSOR_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/preprocessor.pkl"
RESULTS_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/results/model_performance.csv"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

def load_data():
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(INPUT_PATH)
        print(f"✅ Successfully loaded dataset from {INPUT_PATH}")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {INPUT_PATH}")
        exit()

def clean_floor_level(df):
    """Convert textual floor levels into numerical values"""
    mapping = {
        "ground": 0,
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "penthouse": 10
    }
    df["floor_level"] = df["floor_level"].astype(str).str.lower().map(mapping).fillna(df["floor_level"])
    df["floor_level"] = pd.to_numeric(df["floor_level"], errors="coerce")
    return df

def preprocess_data(df):
    """Preprocess the dataset: encode categorical variables and scale numerical ones."""

    df = clean_floor_level(df)  

    categorical_features = ['city', 'neighborhood', 'property_type', 'heating_type']
    numerical_features = ['square_meters', 'num_bedrooms', 'num_bathrooms', 
                          'floor_level', 'distance_to_center_km', 'house_age']

    y = df['price_eur']
    X = df.drop(columns=['price_eur', 'year_built', 'price_per_sq_meter'], errors='ignore')

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(str)

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)

    return X_transformed, y, preprocessor

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

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    return best_model, results

if __name__ == "__main__":
    df = load_data()
    
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(RESULTS_PATH)

    print(f"✅ Model training complete. Best model saved at: {MODEL_PATH}")
    print(f"🔹 Preprocessor saved at: {PREPROCESSOR_PATH}")
    print(f"📊 Model performance saved at: {RESULTS_PATH}")
