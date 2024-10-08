import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    # Load dataset (example path)
    df = pd.read_csv('data/processed/final_binned_global.csv')

    # Split into features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
