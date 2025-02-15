import pandas as pd
import os

# Define paths
INPUT_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/processed_data.csv"
OUTPUT_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/featured_data.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def load_data():
    """Load dataset."""
    return pd.read_csv(INPUT_PATH)

def create_new_features(df):
    """Generate new features and fix categorical encoding issues."""

    # Convert categorical features to strings
    categorical_features = ['city', 'neighborhood', 'property_type', 'heating_type', 'floor_level']
    df[categorical_features] = df[categorical_features].astype(str)

    # Convert `floor_level` into numerical categories
    df['floor_level'] = df['floor_level'].map({
        'ground': 0,
        'first': 1,
        'second': 2,
        'third': 3,
        'fourth': 4
    }).fillna(-1)  # Set missing values to -1

    # Create additional numerical features
    df['price_per_sq_meter'] = df['price_eur'] / df['square_meters']
    df['house_age'] = 2025 - df['year_built']

    return df

if __name__ == "__main__":
    df = load_data()
    df = create_new_features(df)

    # Save updated dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Feature engineering complete. Data saved at: {OUTPUT_PATH}")
