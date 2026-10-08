"""
Model Training and Cross-Validation Pipeline
Machine Learning Assignment 1: Polynomial Regression
Author: Aditya Mittal (BT2024188)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate

def evaluate_models():
    print("="*65)
    print("MACHINE LEARNING ASSIGNMENT 1: MODEL TRAINING & CROSS-VALIDATION")
    print("Student Roll Number: BT2024188")
    print("="*65)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # -------------------------------------------------------------
    # PHASE 1: STEAM TURBINE OPTIMIZATION (var1)
    # -------------------------------------------------------------
    print("\n" + "-"*30 + " PHASE 1 (var1) " + "-"*30)
    tr1 = pd.read_csv("BT2024188_train_var1.csv")
    X1 = tr1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    y1 = tr1['y'].values
    
    print(f"{'Degree':<8}{'Terms':<8}{'OLS Tr MSE':<12}{'OLS Val MSE':<13}{'OLS Val R2':<12}{'Ridge Val MSE':<15}{'Ridge Val R2':<13}{'Alpha'}")
    for d in [1, 2, 3, 4, 5]:
        poly = PolynomialFeatures(degree=d, include_bias=True)
        m_ols = Pipeline([('poly', poly), ('lr', LinearRegression(fit_intercept=False))])
        cv_o = cross_validate(m_ols, X1, y1, cv=kf, scoring=['neg_mean_squared_error', 'r2'], return_train_score=True)
        
        tr_mse_o = -cv_o['train_neg_mean_squared_error'].mean()
        val_mse_o = -cv_o['test_neg_mean_squared_error'].mean()
        val_r2_o = cv_o['test_r2'].mean()
        n_terms = poly.fit(X1).n_output_features_
        
        alpha = 2.0 if d == 5 else (5.0 if d == 4 else 1.0)
        m_r = Pipeline([('poly', poly), ('ridge', Ridge(alpha=alpha, fit_intercept=False, random_state=42))])
        cv_r = cross_validate(m_r, X1, y1, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
        val_mse_r = -cv_r['test_neg_mean_squared_error'].mean()
        val_r2_r = cv_r['test_r2'].mean()
        
        print(f"{d:<8}{n_terms:<8}{tr_mse_o:<12.4f}{val_mse_o:<13.4f}{val_r2_o:<12.4f}{val_mse_r:<15.4f}{val_r2_r:<13.4f}{alpha:<6.1f}")
        
    print("\nPhase 1 Optimal Decision:")
    print("  - Without Regularization: Degree 4 OLS (Val MSE: 0.8978, Val R2: 0.9150)")
    print("  - With Regularization:    Degree 5 Ridge [alpha=2.0] (Val MSE: 0.4589, Val R2: 0.9564)")
    
    # -------------------------------------------------------------
    # PHASE 2: SUBTERRANEAN THERMAL RESERVOIR MAPPING (var2)
    # -------------------------------------------------------------
    print("\n" + "-"*30 + " PHASE 2 (var2) " + "-"*30)
    tr2 = pd.read_csv("BT2024188_train_var2.csv")
    X2 = tr2[['x1', 'x2', 'x3']].values
    y2 = tr2['y'].values
    
    print(f"{'Degree':<8}{'Terms':<8}{'OLS Tr MSE':<12}{'OLS Val MSE':<13}{'OLS Val R2':<12}{'Ridge Val MSE':<15}{'Ridge Val R2':<13}{'Alpha'}")
    for d in range(1, 11):
        poly = PolynomialFeatures(degree=d, include_bias=True)
        m_ols = Pipeline([('poly', poly), ('lr', LinearRegression(fit_intercept=False))])
        cv_o = cross_validate(m_ols, X2, y2, cv=kf, scoring=['neg_mean_squared_error', 'r2'], return_train_score=True)
        
        tr_mse_o = -cv_o['train_neg_mean_squared_error'].mean()
        val_mse_o = -cv_o['test_neg_mean_squared_error'].mean()
        val_r2_o = cv_o['test_r2'].mean()
        n_terms = poly.fit(X2).n_output_features_
        
        alpha = 0.05 if d == 10 else 0.01
        m_r = Pipeline([('poly', poly), ('ridge', Ridge(alpha=alpha, fit_intercept=False, random_state=42))])
        cv_r = cross_validate(m_r, X2, y2, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
        val_mse_r = -cv_r['test_neg_mean_squared_error'].mean()
        val_r2_r = cv_r['test_r2'].mean()
        
        print(f"{d:<8}{n_terms:<8}{tr_mse_o:<12.4f}{val_mse_o:<13.4f}{val_r2_o:<12.4f}{val_mse_r:<15.4f}{val_r2_r:<13.4f}{alpha:<6.2f}")
        
    print("\nPhase 2 Optimal Decision:")
    print("  - Without Regularization: Degree 8 OLS (Val MSE: 0.2466, Val R2: 0.9946)")
    print("  - With Regularization:    Degree 8 Ridge [alpha=0.01] (Val MSE: 0.2409, Val R2: 0.9948)")
    print("  - Peak R2 Ridge:          Degree 10 Ridge [alpha=0.05] (Val MSE: 0.2272, Val R2: 0.9951)")

if __name__ == "__main__":
    evaluate_models()
