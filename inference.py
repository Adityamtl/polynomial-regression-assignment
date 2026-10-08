"""
Inference Script for Generating Test Predictions
Machine Learning Assignment 1: Polynomial Regression
Author: Aditya Mittal (BT2024188)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def run_inference():
    print("="*65)
    print("RUNNING INFERENCE & GENERATING SUBMISSION PREDICTION CSVs")
    print("Student Roll Number: BT2024188")
    print("="*65)
    
    # 1. Load Training and Test Datasets
    tr1 = pd.read_csv("BT2024188_train_var1.csv")
    te1 = pd.read_csv("BT2024188_test_var1.csv")
    X1_train = tr1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    y1_train = tr1['y'].values
    X1_test = te1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    
    tr2 = pd.read_csv("BT2024188_train_var2.csv")
    te2 = pd.read_csv("BT2024188_test_var2.csv")
    X2_train = tr2[['x1', 'x2', 'x3']].values
    y2_train = tr2['y'].values
    X2_test = te2[['x1', 'x2', 'x3']].values
    
    # 2. Phase 1 Model (Degree 5 Ridge Regression, alpha=2.0)
    print("\nTraining Phase 1 Model: Polynomial Degree 5 + Ridge (alpha=2.0)...")
    model_p1 = Pipeline([
        ('poly', PolynomialFeatures(degree=5, include_bias=True)),
        ('ridge', Ridge(alpha=2.0, fit_intercept=False, random_state=42))
    ])
    model_p1.fit(X1_train, y1_train)
    pred_p1 = model_p1.predict(X1_test)
    
    # 3. Phase 2 Model (Degree 8 Ridge Regression, alpha=0.01)
    print("Training Phase 2 Model: Polynomial Degree 8 + Ridge (alpha=0.01)...")
    model_p2 = Pipeline([
        ('poly', PolynomialFeatures(degree=8, include_bias=True)),
        ('ridge', Ridge(alpha=0.01, fit_intercept=False, random_state=42))
    ])
    model_p2.fit(X2_train, y2_train)
    pred_p2 = model_p2.predict(X2_test)
    
    # 4. Save to CSV
    out_file1 = "BT2024188_pred_var1.csv"
    out_file2 = "BT2024188_pred_var2.csv"
    
    df_p1 = pd.DataFrame({'y': pred_p1})
    df_p2 = pd.DataFrame({'y': pred_p2})
    
    df_p1.to_csv(out_file1, index=False)
    df_p2.to_csv(out_file2, index=False)
    
    print(f"\n[SUCCESS] Wrote {out_file1} (Shape: {df_p1.shape})")
    print(f"[SUCCESS] Wrote {out_file2} (Shape: {df_p2.shape})")
    
    # 5. Sanity Checks against sample_submission.csv
    sample = pd.read_csv("sample_submission.csv")
    for name, df in [(out_file1, df_p1), (out_file2, df_p2)]:
        assert df.shape == sample.shape, f"Shape mismatch in {name}!"
        assert list(df.columns) == ['y'], f"Column mismatch in {name}!"
        assert not df['y'].isnull().any(), f"NaN values detected in {name}!"
        assert not np.isinf(df['y']).any(), f"Infinite values detected in {name}!"
        
    print("\n[VALIDATION] Both prediction CSVs match sample_submission.csv format perfectly!")

if __name__ == "__main__":
    run_inference()
