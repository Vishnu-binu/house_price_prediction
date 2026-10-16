
import os
import joblib
import pandas as pd

from flask import Flask, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")

REGRESSION_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
CLASSIFICATION_PATH = os.path.join(MODEL_DIR, "classification_model.pkl")

app = Flask(__name__)

FIELD_LIMITS = {
    "longitude": (-124.35, -114.31),
    "latitude": (32.54, 41.95),
    "housing_median_age": (1, 52),
    "total_rooms": (2, 39320),
    "total_bedrooms": (1, 6445),
    "households": (1, 6082),
}

OCEAN_OPTIONS = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
]


def load_models():
    if not os.path.exists(REGRESSION_PATH):
        raise FileNotFoundError(
            "Regression model missing. Run python train_model.py first."
        )

    if not os.path.exists(CLASSIFICATION_PATH):
        raise FileNotFoundError(
            "Classification model missing. Run python train_model.py first."
        )

    regression_data = joblib.load(REGRESSION_PATH)
    classification_data = joblib.load(CLASSIFICATION_PATH)

    return regression_data, classification_data


def get_hidden_defaults():
    """Supply median values for fields hidden from the form."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "data/housing.csv was not found."
        )

    data = pd.read_csv(DATA_PATH)

    defaults = {}

    for feature in ["population", "median_income"]:
        if feature not in data.columns:
            raise ValueError(f"Dataset is missing {feature}.")

        defaults[feature] = float(
            pd.to_numeric(data[feature], errors="coerce").median()
        )

    return defaults


def get_category(price, low_limit, high_limit):
    if price <= low_limit:
        return "Low"
    if price <= high_limit:
        return "Medium"
    return "High"


def render_page(
    prediction=None,
    category=None,
    error=None,
    form_data=None,
):
    form_data = form_data or {}

    try:
        regression_data, classification_data = load_models()

        model_name = regression_data.get(
            "best_model_name", "Trained Model"
        )
        r2_score_value = regression_data.get("r2_score")
        accuracy = classification_data.get("accuracy")
        low_limit = classification_data.get("low_limit")
        high_limit = classification_data.get("high_limit")

    except (FileNotFoundError, KeyError, ValueError) as exc:
        model_name = "Not available"
        r2_score_value = None
        accuracy = None
        low_limit = None
        high_limit = None

        if error is None:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        category=category,
        error=error,
        form_data=form_data,
        field_limits=FIELD_LIMITS,
        ocean_options=OCEAN_OPTIONS,
        model_name=model_name,
        r2_score_value=r2_score_value,
        accuracy=accuracy,
        low_limit=low_limit,
        high_limit=high_limit,
    )


@app.route("/")
def home():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    try:
        regression_data, classification_data = load_models()

        model = regression_data["model"]
        features = regression_data["features"]

        defaults = get_hidden_defaults()
        input_values = {}

        for feature in features:
            # These two fields are intentionally hidden in the webpage.
            if feature in defaults:
                input_values[feature] = defaults[feature]
                continue

            if feature == "ocean_proximity":
                value = request.form.get(feature, "").strip()

                if value not in OCEAN_OPTIONS:
                    return render_page(
                        error="Please select a valid ocean proximity.",
                        form_data=form_data,
                    )

                input_values[feature] = value
                continue

            raw_value = request.form.get(feature, "").strip()

            if not raw_value:
                return render_page(
                    error=f"Please enter {feature.replace('_', ' ')}.",
                    form_data=form_data,
                )

            try:
                value = float(raw_value)
            except ValueError:
                return render_page(
                    error=f"Invalid number for {feature.replace('_', ' ')}.",
                    form_data=form_data,
                )

            if feature in FIELD_LIMITS:
                minimum, maximum = FIELD_LIMITS[feature]

                if not minimum <= value <= maximum:
                    return render_page(
                        error=(
                            f"{feature.replace('_', ' ').title()} "
                            f"must be between {minimum} and {maximum}."
                        ),
                        form_data=form_data,
                    )

            input_values[feature] = value

        input_df = pd.DataFrame([input_values], columns=features)

        predicted_price = float(model.predict(input_df)[0])
        predicted_price = max(0.0, predicted_price)

        category = get_category(
            predicted_price,
            classification_data["low_limit"],
            classification_data["high_limit"],
        )

        return render_page(
            prediction=predicted_price,
            category=category,
            form_data=form_data,
        )

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return render_page(
            error=f"Prediction failed: {exc}",
            form_data=form_data,
        )


if __name__ == "__main__":
    app.run(debug=True)