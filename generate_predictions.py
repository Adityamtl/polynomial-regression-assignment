import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Load datasets
tr1 = pd.read_csv("BT2024188_train_var1.csv")
X1_train = tr1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1_train = tr1['y'].values
te1 = pd.read_csv("BT2024188_test_var1.csv")
X1_test = te1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values

tr2 = pd.read_csv("BT2024188_train_var2.csv")
X2_train = tr2[['x1', 'x2', 'x3']].values
y2_train = tr2['y'].values
te2 = pd.read_csv("BT2024188_test_var2.csv")
X2_test = te2[['x1', 'x2', 'x3']].values

# Phase 1 Model: Degree 5 Polynomial with Ridge (alpha=2.0)
model_var1 = Pipeline([
    ('poly', PolynomialFeatures(degree=5, include_bias=True)),
    ('ridge', Ridge(alpha=2.0, fit_intercept=False, random_state=42))
])
model_var1.fit(X1_train, y1_train)
pred_var1 = model_var1.predict(X1_test)

# Phase 2 Model: Degree 8 Polynomial with Ridge (alpha=0.01)
model_var2 = Pipeline([
    ('poly', PolynomialFeatures(degree=8, include_bias=True)),
    ('ridge', Ridge(alpha=0.01, fit_intercept=False, random_state=42))
])
model_var2.fit(X2_train, y2_train)
pred_var2 = model_var2.predict(X2_test)

# Create DataFrames
df_pred1 = pd.DataFrame({'y': pred_var1})
df_pred2 = pd.DataFrame({'y': pred_var2})

# Save to CSV
pred1_path = "BT2024188_pred_var1.csv"
pred2_path = "BT2024188_pred_var2.csv"
df_pred1.to_csv(pred1_path, index=False)
df_pred2.to_csv(pred2_path, index=False)

print(f"Generated {pred1_path} with shape {df_pred1.shape}")
print(f"Generated {pred2_path} with shape {df_pred2.shape}")

# Verify against sample_submission.csv
sample = pd.read_csv("sample_submission.csv")
print("Sample submission shape:", sample.shape)
print("Sample submission columns:", list(sample.columns))

assert df_pred1.shape == sample.shape, "Shape mismatch for var1!"
assert df_pred2.shape == sample.shape, "Shape mismatch for var2!"
assert list(df_pred1.columns) == ['y'], "Column mismatch for var1!"
assert list(df_pred2.columns) == ['y'], "Column mismatch for var2!"
assert not df_pred1['y'].isnull().any(), "NaN found in pred1!"
assert not df_pred2['y'].isnull().any(), "NaN found in pred2!"

print("ALL SUBMISSION VALIDATION CHECKS PASSED!")
