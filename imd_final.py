#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Required Libraries

# In[ ]:


import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

print("Libraries imported successfully.")


# ## 2. Load Dataset

# In[ ]:


DATA_PATH =  "/Users/alismac/Documents/Monsoon Hackthon Project/Indian Rainfall Dataset District-wise Daily Measurements.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please place the CSV file in the same folder as this notebook "
        "or update DATA_PATH with the correct file path."
    )


data = pd.read_csv(DATA_PATH, sep=None, engine="python")

print("Dataset loaded successfully.")
print("Shape:", data.shape)

data.head()


# ## 3. Basic Data Understanding

# In[ ]:


print("Columns:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())


# In[ ]:


data.describe(include="all").T


# ## 4. Clean Column Names

# In[ ]:


data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print(data.columns.tolist())


# ## 5. Detect Daily Rainfall Columns

# In[ ]:


def get_day_suffix(day):
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


possible_day_cols = []

for day in range(1, 32):
    possible_day_cols.extend([
        str(day),
        f"{day}{get_day_suffix(day)}",
        f"day_{day}",
        f"d{day}"
    ])

day_cols = [col for col in possible_day_cols if col in data.columns]

print("Detected day columns:", day_cols)
print("Number of day columns:", len(day_cols))

if len(day_cols) == 0:
    raise ValueError(
        "No daily rainfall columns were detected. "
        "Please check the column names in the dataset."
    )


# ## 6. Handle Missing Values and Create Target Column

# In[ ]:


for col in day_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data[day_cols] = data[day_cols].fillna(0)

# Monthly total rainfall is the target variable.
data["total_rainfall"] = data[day_cols].sum(axis=1)

data[["total_rainfall"]].head()


# ## 7. Exploratory Data Analysis

# In[ ]:


monthly_rainfall = data.groupby("month")["total_rainfall"].mean()

plt.figure(figsize=(8, 4))
plt.plot(monthly_rainfall.index, monthly_rainfall.values, marker="o")
plt.title("Average Rainfall by Month")
plt.xlabel("Month")
plt.ylabel("Average Total Rainfall")
plt.grid(True)
plt.show()


# In[ ]:


state_rainfall = (
    data.groupby("state")["total_rainfall"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
state_rainfall.plot(kind="bar")
plt.title("Top 10 States by Total Rainfall")
plt.xlabel("State")
plt.ylabel("Total Rainfall")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


district_rainfall = (
    data.groupby("district")["total_rainfall"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
district_rainfall.plot(kind="bar")
plt.title("Top 10 Districts by Total Rainfall")
plt.xlabel("District")
plt.ylabel("Total Rainfall")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ## 8. Prepare Data for Machine Learning
# 

# In[ ]:


required_cols = ["state", "district", "month", "total_rainfall"]

missing_required_cols = [col for col in required_cols if col not in data.columns]

if missing_required_cols:
    raise ValueError(f"Missing required columns: {missing_required_cols}")

model_data = data[required_cols].copy()

X = model_data[["state", "district", "month"]]
y = model_data["total_rainfall"]

X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ## 9. Build Preprocessing Pipeline

# In[ ]:


categorical_features = ["state", "district"]
numeric_features = ["month"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features)
    ]
)

preprocessor
#categorical_features


# ## 10. Train and Compare Models

# In[ ]:


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": RandomForestRegressor(
        n_estimators=1,
        max_depth=8,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

trained_models = {}

for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[model_name] = pipeline

results_df = pd.DataFrame(results).sort_values(by="RMSE")
results_df


# In[ ]:


best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("Best Model:", best_model_name)


# ## 11. Actual vs Predicted Values

# In[ ]:


best_predictions = best_model.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": best_predictions
})

comparison_df.head(10)


# In[ ]:


plt.figure(figsize=(6, 6))
plt.scatter(comparison_df["Actual"], comparison_df["Predicted"], alpha=0.6)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title(f"Actual vs Predicted Rainfall - {best_model_name}")
plt.grid(True)
plt.show()


# ## 12. Prediction Function

# In[ ]:


def predict_rainfall(state, district, month):
    input_data = pd.DataFrame({
        "state": [state],
        "district": [district],
        "month": [month]
    })

    prediction = best_model.predict(input_data)[0]
    return prediction


# Example prediction
example_state = data["state"].iloc[0]
example_district = data["district"].iloc[0]
example_month = int(data["month"].iloc[0])

predicted_rainfall = predict_rainfall(example_state, example_district, example_month)

print("State:", example_state)
print("District:", example_district)
print("Month:", example_month)
print("Predicted Rainfall:", round(predicted_rainfall, 2))


# 
