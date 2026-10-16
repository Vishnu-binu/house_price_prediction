
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# 1. FOLDERS AND FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
CLASSIFICATION_PATH = os.path.join(
    MODEL_DIR, "classification_model.pkl"
)

CLEANED_DATA_PATH = os.path.join(
    BASE_DIR, "data", "cleaned_data.csv"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR, "model_comparison.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    STATIC_DIR, "confusion_matrix.png"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)


# =========================================================
# 2. LOAD DATASET
# =========================================================

print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place housing.csv inside the data folder."
    )

data = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())


# =========================================================
# 3. DATA CLEANING
# =========================================================

required_columns = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
    "median_house_value",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

data = data[required_columns].copy()

# Remove duplicate rows
data = data.drop_duplicates()

# Convert numeric columns to numeric types
numeric_columns = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column], errors="coerce"
    )

# Remove rows without a valid target value
data = data.dropna(subset=["median_house_value"])

# Remove invalid target values
data = data[data["median_house_value"] > 0]

# Remove rows without the categorical value
data = data.dropna(subset=["ocean_proximity"])

if len(data) < 10:
    raise ValueError(
        "Not enough valid rows to train the models."
    )

# Save cleaned dataset
data.to_csv(CLEANED_DATA_PATH, index=False)

print("\nCleaned Dataset Shape:", data.shape)
print("Cleaned dataset saved:", CLEANED_DATA_PATH)


# =========================================================
# 4. FEATURES AND TARGET
# =========================================================

TARGET = "median_house_value"

X = data.drop(columns=[TARGET])
y = data[TARGET]

numeric_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

categorical_features = ["ocean_proximity"]

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Column:", TARGET)


# =========================================================
# 5. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining Rows:", len(X_train))
print("Testing Rows:", len(X_test))


# =========================================================
# 6. PREPROCESSING
# =========================================================

def create_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )


# =========================================================
# 7. REGRESSION MODELS
# =========================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        random_state=42,
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=3,
    ),
}


# =========================================================
# 8. TRAIN AND EVALUATE REGRESSION MODELS
# =========================================================

results = []
trained_models = {}

print("\n" + "=" * 60)
print("REGRESSION MODEL EVALUATION")
print("=" * 60)

for name, regressor in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", clone(regressor)),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    trained_models[name] = pipeline

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    print(f"MAE      : ${mae:,.2f}")
    print(f"MSE      : {mse:,.2f}")
    print(f"RMSE     : ${rmse:,.2f}")
    print(f"R2 Score : {r2:.4f}")


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False,
)

results_df.to_csv(COMPARISON_PATH, index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nComparison saved:", COMPARISON_PATH)


# =========================================================
# 10. SELECT AND SAVE BEST REGRESSION MODEL
# =========================================================

best_model_name = results_df.iloc[0]["Model"]
best_r2 = float(results_df.iloc[0]["R2"])

best_model = trained_models[best_model_name]

regression_data = {
    "model": best_model,
    "scaler": None,
    "scaled": False,
    "features": list(X.columns),
    "target": TARGET,
    "best_model_name": best_model_name,
    "r2_score": best_r2,
}

joblib.dump(regression_data, MODEL_PATH)

print("\nBest Regression Model:", best_model_name)
print("Best Test R2 Score:", round(best_r2, 4))
print("Regression model saved:", MODEL_PATH)


# =========================================================
# 11. CREATE PRICE CATEGORIES
# =========================================================

print("\n" + "=" * 60)
print("PRICE CATEGORY CLASSIFICATION")
print("=" * 60)

# Calculate category limits from training targets only
low_limit = float(y_train.quantile(0.33))
high_limit = float(y_train.quantile(0.66))


def price_category(price):
    if price <= low_limit:
        return "Low"
    elif price <= high_limit:
        return "Medium"
    else:
        return "High"


y_train_class = y_train.apply(price_category)
y_test_class = y_test.apply(price_category)

print(f"Low category limit    : ${low_limit:,.2f}")
print(f"Medium category limit : ${high_limit:,.2f}")
print(f"High category         : Above ${high_limit:,.2f}")


# =========================================================
# 12. TRAIN CLASSIFICATION MODEL
# =========================================================

classifier_pipeline = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        ),
    ]
)

print("\nTraining Random Forest Classifier...")

classifier_pipeline.fit(X_train, y_train_class)

class_predictions = classifier_pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test_class,
    class_predictions,
)

labels = ["Low", "Medium", "High"]

cm = confusion_matrix(
    y_test_class,
    class_predictions,
    labels=labels,
)


# =========================================================
# 13. CLASSIFICATION RESULTS
# =========================================================

print("\nClassification Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test_class,
        class_predictions,
        labels=labels,
        zero_division=0,
    )
)


# =========================================================
# 14. SAVE CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)

plt.title("House Price Category Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("\nConfusion matrix saved:", CONFUSION_MATRIX_PATH)


# =========================================================
# 15. SAVE CLASSIFICATION MODEL
# =========================================================

classification_data = {
    "classifier": classifier_pipeline,
    "low_limit": low_limit,
    "high_limit": high_limit,
    "accuracy": accuracy,
    "labels": labels,
}

joblib.dump(
    classification_data,
    CLASSIFICATION_PATH,
)

print("Classification model saved:", CLASSIFICATION_PATH)


# =========================================================
# 16. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("PROJECT TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Best Regression Model :", best_model_name)
print(f"Test R2 Score         : {best_r2:.4f}")
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

print("\nGenerated Files:")
print("1. data/cleaned_data.csv")
print("2. model/house_price_model.pkl")
print("3. model/classification_model.pkl")
print("4. model_comparison.csv")
print("5. static/confusion_matrix.png")

print("\nNOTE: House values are in USD, not INR.")