import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =========================
# 1️⃣ LOAD DATASET
# =========================
data = pd.read_csv(r"C:\Users\soman\OneDrive\Desktop\Rainfall-Prediction-App-Final\Rainfall.csv")   # <-- put your csv name here

print("\nDataset Loaded Successfully!\n")

# =========================
# 2️⃣ SEE ACTUAL COLUMN NAMES
# =========================
print("Available Columns in Dataset:\n")
print(data.columns.tolist())

# Remove spaces from column names (very important fix)
data.columns = data.columns.str.strip()

# =========================
# 3️⃣ HANDLE NULL VALUES (if any)
# =========================
data = data.fillna(data.mean(numeric_only=True))

# =========================
# 4️⃣ DEFINE TARGET COLUMN
# =========================
# CHANGE this to your rainfall column name if different
target_column = "rainfall"   # <-- must match your dataset

if target_column not in data.columns:
    raise ValueError(f"\n'{target_column}' column not found. Please check dataset.\n")

# =========================
# 5️⃣ DEFINE FEATURES (AUTO SELECT)
# =========================
# Use ALL columns except target as features
feature_cols = [col for col in data.columns if col != target_column]

print("\nSelected Features:\n", feature_cols)

X = data[feature_cols]
y = data[target_column]

# =========================
# 6️⃣ SCALE FEATURES (important)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 7️⃣ TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

print("\nModel Training Completed!")

# =========================
# 8️⃣ SAVE FILES FOR STREAMLIT
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(feature_cols, open("features.pkl", "wb"))

print("\n✅ model.pkl, scaler.pkl, features.pkl saved successfully!")