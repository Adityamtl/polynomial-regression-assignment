import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

ROLLNO = "BT2024188"
DATA_DIR = r"c:\Users\adityamittal\Downloads\New folder (14)"

def check_files():
    req_files = [
        f"{ROLLNO}_train_var1.csv",
        f"{ROLLNO}_test_var1.csv",
        f"{ROLLNO}_train_var2.csv",
        f"{ROLLNO}_test_var2.csv"
    ]
    missing = [f for f in req_files if not os.path.exists(os.path.join(DATA_DIR, f))]
    return missing

def main():
    missing = check_files()
    if missing:
        print(f"MISSING_FILES: {missing}")
        return False
    print("ALL 4 DATASET FILES FOUND! Proceeding with complete pipeline...")
    return True

if __name__ == "__main__":
    main()
