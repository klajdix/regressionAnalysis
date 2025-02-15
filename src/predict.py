import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/best_model.pkl"
PREPROCESSOR_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/models/preprocessor.pkl"

def load_model():
    """Load trained model and preprocessor."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print("❌ ERROR: Model or preprocessor file not found.")
        exit()

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✅ Model and preprocessor loaded successfully.")
    return model, preprocessor

def clean_floor_level(floor):
    """Convert textual floor levels into numerical values"""
    mapping = {
        "ground": 0,
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "penthouse": 10
    }
    return mapping.get(str(floor).lower(), floor)  

def make_prediction(input_data, preprocessor, model):
    """Prepare data and make price prediction."""
    categorical_features = ['city', 'neighborhood', 'property_type', 'heating_type']
    for col in categorical_features:
        input_data[col] = input_data[col].astype(str)

    
    input_data["floor_level"] = input_data["floor_level"].apply(clean_floor_level)

    input_transformed = preprocessor.transform(input_data)

    prediction = model.predict(input_transformed)
    return prediction[0]

if __name__ == "__main__":
    model, preprocessor = load_model()

    sample_data = pd.DataFrame([{
        "city": "5",
        "neighborhood": "0",
        "property_type": "2",
        "square_meters": 96.0,
        "num_bedrooms": 2,
        "num_bathrooms": 1,
        "floor_level": "first",  
        "heating_type": "central",
        "distance_to_center_km": 6.5,
        "house_age": 11.0
    }])

    predicted_price = make_prediction(sample_data, preprocessor, model)

    print(f"🏡 Predicted house price: {predicted_price:,.2f} EUR")
