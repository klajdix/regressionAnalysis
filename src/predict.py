import pandas as pd
import joblib
import numpy as np

# Define paths
MODEL_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/best_model.pkl"
PREPROCESSOR_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/preprocessor.pkl"
FEATURED_DATA_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/featured_data.csv"

# Load model & preprocessor
def load_model():
    """Load trained model from file."""
    return joblib.load(MODEL_PATH)

def load_preprocessor():
    """Load saved preprocessor."""
    return joblib.load(PREPROCESSOR_PATH)

def make_prediction(input_data, preprocessor):
    """Predict house price using the trained model."""
    model = load_model()

    # Apply preprocessing
    input_transformed = preprocessor.transform(input_data)

    prediction = model.predict(input_transformed)
    return prediction

if __name__ == "__main__":
    # Load dataset and sample a row
    df = pd.read_csv(FEATURED_DATA_PATH)

    # Select a sample row for prediction (drop price_eur)
    sample_data = df.drop(columns=['price_eur', 'year_built']).iloc[[0]]  

    # Load preprocessor
    preprocessor = load_preprocessor()

    # Make prediction
    predicted_price = make_prediction(sample_data, preprocessor)

    print(f"🏡 Predicted house price: {predicted_price[0]:,.2f} EUR")
