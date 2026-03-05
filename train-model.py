import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

data = pd.read_csv("dataset.csv")

# remove missing values
data = data.dropna()

# encoders
crop_encoder = LabelEncoder()
state_encoder = LabelEncoder()
season_encoder = LabelEncoder()

data["Crop"] = crop_encoder.fit_transform(data["Crop"])
data["State"] = state_encoder.fit_transform(data["State"])
data["Season"] = season_encoder.fit_transform(data["Season"])

# features
X = data[[
    "Crop",
    "State",
    "Season",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]]

# target
y = data["Yield"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# evaluate
pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, pred))

# save model and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(crop_encoder, open("crop_encoder.pkl", "wb"))
pickle.dump(state_encoder, open("state_encoder.pkl", "wb"))
pickle.dump(season_encoder, open("season_encoder.pkl", "wb"))

print("Training completed.")