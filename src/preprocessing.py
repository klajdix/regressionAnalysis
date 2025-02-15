import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Define file paths
FILE_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/Albania_data_realestate.csv"
OUTPUT_PATH = "/Users/klajdtopuzi/Desktop/DataAnalysis/regressionAnalysis/data/processed_data.csv"

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded dataset from {filepath}")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {filepath}")
        exit()

def handle_missing_values(df):
    """Fill missing values with median for numerical columns."""
    imputer = SimpleImputer(strategy='median')
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

def encode_categorical_features(df, categorical_columns):
    """Encode categorical features using Label Encoding."""
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            print(f"⚠️ WARNING: Column '{col}' not found. Skipping encoding.")
    return df, label_encoders

if __name__ == "__main__":
    df = load_data(FILE_PATH)
    df = handle_missing_values(df)
    df, encoders = encode_categorical_features(df, ['location'])  # Adjust column names if needed
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Data preprocessing complete. Processed data saved at: {OUTPUT_PATH}")
