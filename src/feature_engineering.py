import pandas as pd

INPUT_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/processed_data.csv"
OUTPUT_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/featured_data.csv"

def create_new_features(df):
    """Create additional features for the dataset."""
    if 'year_built' in df.columns:
        df['house_age'] = 2025 - df['year_built']
    else:
        print("⚠️ WARNING: 'year_built' column not found! 'house_age' cannot be created.")
    
    df['price_per_sq_meter'] = df['price_eur'] / df['square_meters']
    
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    
    df = create_new_features(df)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Feature engineering complete. Data saved at: {OUTPUT_PATH}")
