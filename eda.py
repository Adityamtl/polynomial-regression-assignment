"""
Exploratory Data Analysis (EDA) Script
Machine Learning Assignment 1: Polynomial Regression
Author: Aditya Mittal (BT2024188)
"""

import os
import pandas as pd
import numpy as np

def run_eda():
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA) - BT2024188")
    print("="*60)
    
    # 1. Load Phase 1 Data
    tr1 = pd.read_csv("BT2024188_train_var1.csv")
    te1 = pd.read_csv("BT2024188_test_var1.csv")
    print("\n--- PHASE 1: STEAM TURBINE OPTIMIZATION (var1) ---")
    print(f"Train Shape: {tr1.shape}, Test Shape: {te1.shape}")
    print(f"Null values in train: {tr1.isnull().sum().sum()}, test: {te1.isnull().sum().sum()}")
    print("\nFeature Summary Statistics (Train Var1):")
    print(tr1.describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    print("\nPearson Correlation with Target 'y' (Var1):")
    corr1 = tr1.corr()['y'].sort_values()
    print(corr1)
    
    # 2. Load Phase 2 Data
    tr2 = pd.read_csv("BT2024188_train_var2.csv")
    te2 = pd.read_csv("BT2024188_test_var2.csv")
    print("\n--- PHASE 2: SUBTERRANEAN THERMAL MAPPING (var2) ---")
    print(f"Train Shape: {tr2.shape}, Test Shape: {te2.shape}")
    print(f"Null values in train: {tr2.isnull().sum().sum()}, test: {te2.isnull().sum().sum()}")
    print("\nFeature Summary Statistics (Train Var2):")
    print(tr2.describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    print("\nPearson Correlation with Target 'y' (Var2):")
    corr2 = tr2.corr()['y'].sort_values()
    print(corr2)

if __name__ == "__main__":
    run_eda()
