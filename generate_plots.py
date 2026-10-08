import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict, cross_validate

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

plots_dir = "report_figures"
os.makedirs(plots_dir, exist_ok=True)

# Load data
tr1 = pd.read_csv("BT2024188_train_var1.csv")
X1 = tr1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1 = tr1['y'].values

tr2 = pd.read_csv("BT2024188_train_var2.csv")
X2 = tr2[['x1', 'x2', 'x3']].values
y2 = tr2['y'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 1. PHASE 1 MODEL SELECTION (DEGREE 1 to 5)
# ==========================================
deg1_list = [1, 2, 3, 4, 5]
p1_ols_train_mse, p1_ols_val_mse, p1_ols_val_r2 = [], [], []
p1_ridge_train_mse, p1_ridge_val_mse, p1_ridge_val_r2 = [], [], []

for d in deg1_list:
    poly = PolynomialFeatures(degree=d, include_bias=True)
    # OLS
    m_ols = Pipeline([('poly', poly), ('lr', LinearRegression(fit_intercept=False))])
    cv_o = cross_validate(m_ols, X1, y1, cv=kf, scoring=['neg_mean_squared_error', 'r2'], return_train_score=True)
    p1_ols_train_mse.append(-cv_o['train_neg_mean_squared_error'].mean())
    p1_ols_val_mse.append(-cv_o['test_neg_mean_squared_error'].mean())
    p1_ols_val_r2.append(cv_o['test_r2'].mean())
    
    # Ridge (alpha=2.0 for d=5, alpha=1.0 for others)
    alpha = 2.0 if d == 5 else 1.0
    m_r = Pipeline([('poly', poly), ('ridge', Ridge(alpha=alpha, fit_intercept=False, random_state=42))])
    cv_r = cross_validate(m_r, X1, y1, cv=kf, scoring=['neg_mean_squared_error', 'r2'], return_train_score=True)
    p1_ridge_train_mse.append(-cv_r['train_neg_mean_squared_error'].mean())
    p1_ridge_val_mse.append(-cv_r['test_neg_mean_squared_error'].mean())
    p1_ridge_val_r2.append(cv_r['test_r2'].mean())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(deg1_list, p1_ols_train_mse, 'o--', color='#2b5c8f', label='Train MSE (OLS)', linewidth=1.8)
axes[0].plot(deg1_list, p1_ols_val_mse, 's-', color='#d95f02', label='Validation MSE (OLS)', linewidth=2.2)
axes[0].plot(deg1_list, p1_ridge_val_mse, '^-.', color='#2ca02c', label='Validation MSE (Ridge)', linewidth=2.2)
axes[0].axvline(x=4, color='#7570b3', linestyle=':', label='Optimal Degree (OLS: 4)')
axes[0].set_xlabel('Polynomial Degree')
axes[0].set_ylabel('Mean Squared Error (MSE)')
axes[0].set_title('Phase 1: MSE vs. Polynomial Degree')
axes[0].set_xticks(deg1_list)
axes[0].set_ylim(0, 10.5)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(deg1_list, p1_ols_val_r2, 's-', color='#d95f02', label='Validation $R^2$ (OLS)', linewidth=2.2)
axes[1].plot(deg1_list, p1_ridge_val_r2, '^-.', color='#2ca02c', label='Validation $R^2$ (Ridge)', linewidth=2.2)
axes[1].axhline(y=0.956, color='#2ca02c', linestyle=':', label='Peak $R^2$ = 0.956 (Ridge Deg 5)')
axes[1].set_xlabel('Polynomial Degree')
axes[1].set_ylabel('Coefficient of Determination ($R^2$)')
axes[1].set_title('Phase 1: Validation $R^2$ Score vs. Degree')
axes[1].set_xticks(deg1_list)
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase1_model_selection.png"), dpi=300)
plt.close()
print("Saved phase1_model_selection.png")


# ==========================================
# 2. PHASE 2 MODEL SELECTION (DEGREE 1 to 10)
# ==========================================
deg2_list = list(range(1, 11))
p2_ols_train_mse, p2_ols_val_mse, p2_ols_val_r2 = [], [], []
p2_ridge_val_mse, p2_ridge_val_r2 = [], []

for d in deg2_list:
    poly = PolynomialFeatures(degree=d, include_bias=True)
    m_ols = Pipeline([('poly', poly), ('lr', LinearRegression(fit_intercept=False))])
    cv_o = cross_validate(m_ols, X2, y2, cv=kf, scoring=['neg_mean_squared_error', 'r2'], return_train_score=True)
    p2_ols_train_mse.append(-cv_o['train_neg_mean_squared_error'].mean())
    p2_ols_val_mse.append(-cv_o['test_neg_mean_squared_error'].mean())
    p2_ols_val_r2.append(cv_o['test_r2'].mean())
    
    alpha = 0.01 if d <= 8 else 0.05
    m_r = Pipeline([('poly', poly), ('ridge', Ridge(alpha=alpha, fit_intercept=False, random_state=42))])
    cv_r = cross_validate(m_r, X2, y2, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
    p2_ridge_val_mse.append(-cv_r['test_neg_mean_squared_error'].mean())
    p2_ridge_val_r2.append(cv_r['test_r2'].mean())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(deg2_list, p2_ols_train_mse, 'o--', color='#2b5c8f', label='Train MSE (OLS)', linewidth=1.8)
axes[0].plot(deg2_list, p2_ols_val_mse, 's-', color='#d95f02', label='Validation MSE (OLS)', linewidth=2.2)
axes[0].plot(deg2_list, p2_ridge_val_mse, '^-.', color='#2ca02c', label='Validation MSE (Ridge)', linewidth=2.2)
axes[0].axvline(x=8, color='#7570b3', linestyle=':', label='Optimal Degree (OLS: 8)')
axes[0].set_xlabel('Polynomial Degree')
axes[0].set_ylabel('Mean Squared Error (MSE)')
axes[0].set_title('Phase 2: MSE vs. Polynomial Degree')
axes[0].set_xticks(deg2_list)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(deg2_list, p2_ols_val_r2, 's-', color='#d95f02', label='Validation $R^2$ (OLS)', linewidth=2.2)
axes[1].plot(deg2_list, p2_ridge_val_r2, '^-.', color='#2ca02c', label='Validation $R^2$ (Ridge)', linewidth=2.2)
axes[1].axhline(y=0.995, color='#2ca02c', linestyle=':', label='Peak $R^2$ = 0.995 (Deg 8-10)')
axes[1].set_xlabel('Polynomial Degree')
axes[1].set_ylabel('Coefficient of Determination ($R^2$)')
axes[1].set_title('Phase 2: Validation $R^2$ Score vs. Degree')
axes[1].set_xticks(deg2_list)
axes[1].set_ylim(0.2, 1.05)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase2_model_selection.png"), dpi=300)
plt.close()
print("Saved phase2_model_selection.png")


# ==========================================
# 3. RESIDUALS & PREDICTED VS ACTUAL
# ==========================================
# Best model 1: Degree 5 Ridge(alpha=2.0)
m1_best = Pipeline([('poly', PolynomialFeatures(degree=5)), ('ridge', Ridge(alpha=2.0, fit_intercept=False))])
y1_oof = cross_val_predict(m1_best, X1, y1, cv=kf)
res1 = y1 - y1_oof

# Best model 2: Degree 8 Ridge(alpha=0.01)
m2_best = Pipeline([('poly', PolynomialFeatures(degree=8)), ('ridge', Ridge(alpha=0.01, fit_intercept=False))])
y2_oof = cross_val_predict(m2_best, X2, y2, cv=kf)
res2 = y2 - y2_oof

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# P1 Pred vs Actual
axes[0, 0].scatter(y1, y1_oof, alpha=0.5, color='#2b5c8f', edgecolors='none', s=25)
p1_min, p1_max = min(y1.min(), y1_oof.min()), max(y1.max(), y1_oof.max())
axes[0, 0].plot([p1_min, p1_max], [p1_min, p1_max], 'r--', linewidth=1.8, label='Ideal Fit ($y=\hat{y}$)')
axes[0, 0].set_xlabel('True Net Power Score ($y$)')
axes[0, 0].set_ylabel('Predicted Net Power Score ($\hat{y}$)')
axes[0, 0].set_title('Phase 1: Predicted vs. Actual (5-Fold CV)')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# P1 Residual distribution
sns.histplot(res1, kde=True, ax=axes[0, 1], color='#2b5c8f', bins=30)
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Residual ($y - \hat{y}$)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Phase 1: Residual Error Distribution')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# P2 Pred vs Actual
axes[1, 0].scatter(y2, y2_oof, alpha=0.5, color='#2ca02c', edgecolors='none', s=25)
p2_min, p2_max = min(y2.min(), y2_oof.min()), max(y2.max(), y2_oof.max())
axes[1, 0].plot([p2_min, p2_max], [p2_min, p2_max], 'r--', linewidth=1.8, label='Ideal Fit ($y=\hat{y}$)')
axes[1, 0].set_xlabel('True Thermal Anomaly Score ($y$)')
axes[1, 0].set_ylabel('Predicted Thermal Anomaly Score ($\hat{y}$)')
axes[1, 0].set_title('Phase 2: Predicted vs. Actual (5-Fold CV)')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# P2 Residual distribution
sns.histplot(res2, kde=True, ax=axes[1, 1], color='#2ca02c', bins=30)
axes[1, 1].axvline(0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Residual ($y - \hat{y}$)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Phase 2: Residual Error Distribution')
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "residuals_and_fit.png"), dpi=300)
plt.close()
print("Saved residuals_and_fit.png")


# ==========================================
# 4. EMPIRICAL VS AI-TRAP COMPARISON BAR CHART
# ==========================================
# Calculate trap metrics
trap1_cv = cross_validate(LinearRegression(), PolynomialFeatures(3).fit_transform(X1[:, :3]), y1, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
trap1_mse = -trap1_cv['test_neg_mean_squared_error'].mean()
trap1_r2 = trap1_cv['test_r2'].mean()

trap2_cv = cross_validate(LinearRegression(), PolynomialFeatures(4).fit_transform(X2[:, :1]), y2, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
trap2_mse = -trap2_cv['test_neg_mean_squared_error'].mean()
trap2_r2 = trap2_cv['test_r2'].mean()

opt1_mse = p1_ridge_val_mse[4] # deg 5 ridge
opt1_r2 = p1_ridge_val_r2[4]
opt2_mse = p2_ridge_val_mse[7] # deg 8 ridge
opt2_r2 = p2_ridge_val_r2[7]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
models = ['Phase 1\nAI Trap\n(Deg 3, 3 Feats)', 'Phase 1\nOptimal\n(Deg 5, 6 Feats)', 'Phase 2\nAI Trap\n(Deg 4, 1 Feat)', 'Phase 2\nOptimal\n(Deg 8, 3 Feats)']
mses = [trap1_mse, opt1_mse, trap2_mse, opt2_mse]
r2s = [trap1_r2, opt1_r2, trap2_r2, opt2_r2]
colors = ['#e41a1c', '#377eb8', '#e41a1c', '#4daf4a']

bars0 = axes[0].bar(models, mses, color=colors, width=0.55, edgecolor='black', alpha=0.85)
axes[0].set_ylabel('Validation MSE (Lower is Better)')
axes[0].set_title('Cross-Validation MSE: Trap vs. Optimal')
axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars0:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

bars1 = axes[1].bar(models, r2s, color=colors, width=0.55, edgecolor='black', alpha=0.85)
axes[1].set_ylabel('Validation $R^2$ Score (Higher is Better)')
axes[1].set_title('Validation $R^2$ Score: Trap vs. Optimal')
axes[1].set_ylim(0, 1.15)
axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars1:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "trap_vs_optimal_comparison.png"), dpi=300)
plt.close()
print("Saved trap_vs_optimal_comparison.png")
