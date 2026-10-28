import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_dataset():
    file_path = "Student Depression Dataset.csv"

    data = pd.read_csv(file_path)

    # Remove completely empty rows
    data = data.dropna(how="all")

    # Remove ID because it is not useful for prediction
    if "id" in data.columns:
        data = data.drop(columns=["id"])

    # Clean column names
    data.columns = data.columns.str.strip()

    return data


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model():

    print("Loading Student Depression dataset...")

    data = load_dataset()

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Number of columns:", len(data.columns))

    # Target column
    target = "Depression"

    # Separate input features and target
    X = data.drop(columns=[target])
    y = data[target]

    print("\nTarget distribution:")
    print(y.value_counts())

    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # --------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # --------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # --------------------------------------------------
    # Combine preprocessing
    # --------------------------------------------------

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # --------------------------------------------------
    # Complete ML Pipeline
    # --------------------------------------------------

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # --------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records :", len(X_test))

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nTraining Student Depression Prediction model...")

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    joblib.dump(model, "student_depression_model.pkl")

    print("\nModel saved as student_depression_model.pkl")

    # --------------------------------------------------
    # Save Metrics
    # --------------------------------------------------

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")

    return accuracy


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    train_model()
