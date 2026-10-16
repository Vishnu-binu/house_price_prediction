
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")
CLEANED_DATA_PATH = os.path.join(
    BASE_DIR, "data", "cleaned_data.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. LOAD DATASET
# ==========================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = pd.read_csv(DATA_PATH)

print("=" * 55)
print("HOUSE PRICE DATA VISUALIZATION")
print("=" * 55)

print("\nDataset Shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())


# ==========================================
# 3. CLEAN DATA FOR VISUALIZATION
# ==========================================

target = "median_house_value"

if target not in data.columns:
    raise ValueError(
        f"Target column '{target}' is missing."
    )

data[target] = pd.to_numeric(
    data[target], errors="coerce"
)

data = data.dropna(subset=[target])
data = data[data[target] > 0]

numeric_columns = data.select_dtypes(
    include="number"
).columns.tolist()

print("\nRows available for visualization:", len(data))


# ==========================================
# 4. HOUSE PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=data,
    x="median_house_value",
    bins=50,
    kde=True,
)

plt.title("Distribution of Median House Values")
plt.xlabel("Median House Value (USD)")
plt.ylabel("Number of Districts")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "price_distribution.png"),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ==========================================
# 5. HOUSE AGE DISTRIBUTION
# ==========================================

if "housing_median_age" in data.columns:
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=data,
        x="housing_median_age",
        bins=30,
        kde=True,
    )

    plt.title("Distribution of Housing Median Age")
    plt.xlabel("Housing Median Age (Years)")
    plt.ylabel("Number of Districts")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "housing_age_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# 6. HOUSE VALUE BY OCEAN PROXIMITY
# ==========================================

if "ocean_proximity" in data.columns:
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=data,
        x="ocean_proximity",
        y="median_house_value",
    )

    plt.title("House Values by Ocean Proximity")
    plt.xlabel("Ocean Proximity")
    plt.ylabel("Median House Value (USD)")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "price_by_ocean_proximity.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# 7. LOCATION MAP: LATITUDE AND LONGITUDE
# ==========================================

required_location_columns = [
    "longitude",
    "latitude",
    "median_house_value",
]

if all(
    column in data.columns
    for column in required_location_columns
):
    # Sample rows to keep the scatter plot readable
    location_data = data.sample(
        n=min(8000, len(data)),
        random_state=42,
    )

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        location_data["longitude"],
        location_data["latitude"],
        c=location_data["median_house_value"],
        s=8,
        alpha=0.55,
        cmap="viridis",
    )

    plt.colorbar(
        scatter,
        label="Median House Value (USD)",
    )

    plt.title("Geographical Distribution of House Values")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "location_price_map.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# 8. MEDIAN INCOME VS HOUSE VALUE
# ==========================================

if "median_income" in data.columns:
    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=data.sample(
            n=min(4000, len(data)),
            random_state=42,
        ),
        x="median_income",
        y="median_house_value",
        alpha=0.45,
        s=18,
    )

    plt.title("Median Income vs Median House Value")
    plt.xlabel("Median Income (Dataset Units)")
    plt.ylabel("Median House Value (USD)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "income_vs_price.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# 9. CORRELATION HEATMAP
# ==========================================

if len(numeric_columns) >= 2:
    correlation = data[numeric_columns].corr()

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
    )

    plt.title("Correlation Between Numeric Features")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# 10. AVERAGE HOUSE VALUE BY OCEAN PROXIMITY
# ==========================================

if "ocean_proximity" in data.columns:
    average_prices = (
        data.groupby("ocean_proximity")["median_house_value"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=average_prices,
        x="ocean_proximity",
        y="median_house_value",
    )

    plt.title("Average House Value by Ocean Proximity")
    plt.xlabel("Ocean Proximity")
    plt.ylabel("Average House Value (USD)")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "average_price_by_ocean.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================
# 11. SAVE CLEANED DATASET
# ==========================================

data.to_csv(CLEANED_DATA_PATH, index=False)


# ==========================================
# 12. FINAL OUTPUT
# ==========================================

print("\nVisualization completed successfully!")
print("\nGraphs saved in:")
print(OUTPUT_DIR)

print("\nGenerated graph files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.endswith(".png"):
        print("-", filename)

print("\nCleaned dataset saved at:")
print(CLEANED_DATA_PATH)