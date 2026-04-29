import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ----------------------------
# Load Data
# ----------------------------
flights = pd.read_csv("travel_capstone/flights.csv")
users = pd.read_csv("travel_capstone/users.csv")
hotels = pd.read_csv("travel_capstone/hotels.csv")


# ----------------------------
# Feature Engineering Flights
# ----------------------------
flights["route"] = flights["from"] + "-" + flights["to"]

def convert_time(t):
    hour = int(t)
    minute = t - hour
    return int(hour * 60) + int(minute * 100)

flights["total_time"] = flights["time"].apply(convert_time)

flights["date"] = pd.to_datetime(flights["date"])
flights["weekday_num"] = flights["date"].dt.weekday
flights["month"] = flights["date"].dt.month
flights["year"] = flights["date"].dt.year
flights["speed"] = flights["distance"] / (flights["total_time"] / 60)

# ----------------------------
# Merge Data
# ----------------------------
df = pd.merge(
    flights,
    users,
    how="left",
    left_on="userCode",
    right_on="code"
)

df = pd.merge(
    df,
    hotels,
    how="left",
    on="travelCode",
    suffixes=("_flight", "_hotel")
)

# ----------------------------
# Fill Missing Values
# ----------------------------
df.fillna(0, inplace=True)

# ----------------------------
# Target Variable
# ----------------------------
target = "price_flight"

# ----------------------------
# Select Useful Columns
# ----------------------------
features = [
    "from", "to", "flightType", "agency",
    "weekday_num", "month", "year", "speed",
    "gender", "age", "company",
    "place", "days", "price_hotel", "total"
]

final_df = df[features + [target]]

# ----------------------------
# Encode Categoricals
# ----------------------------
for col in final_df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    final_df[col] = le.fit_transform(final_df[col].astype(str))

# ----------------------------
# Split Data
# ----------------------------
X = final_df.drop(target, axis=1)
y = final_df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Models
# ----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "LightGBM": LGBMRegressor(),
    "XGBoost": XGBRegressor()
}

mlflow.set_experiment("Travel Price Prediction Multi Dataset")

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(model, name)

        print(name, "logged")