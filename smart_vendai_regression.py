# ============================================================
# Smart VendAI — Regression Modelling for Sales Forecasting
# Author  : Nevin Nelson (Q1071615)
# Course  : BSc. Computer Science and Digitisation (2024-2027)
# Tool    : Google Colab / Python
# Purpose : Predict Units Sold from Foot Traffic data using
#           Simple Linear Regression
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------
# 1. DATASET  (12-month vending machine sales & foot traffic)
# ------------------------------------------------------------
data = {
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    "FootTraffic": [
        1200, 1350, 1500, 1400, 1600, 1750,
        1800, 1700, 1550, 1450, 1300, 1900
    ],
    "UnitsSold": [
        1120, 1260, 1400, 1310, 1490, 1640,
        1680, 1590, 1450, 1350, 1210, 1770
    ]
}

df = pd.DataFrame(data)
print("=== Dataset ===")
print(df.to_string(index=False))
print()

# ------------------------------------------------------------
# 2. SIMPLE LINEAR REGRESSION
# ------------------------------------------------------------
X = df[["FootTraffic"]]   # Independent variable
y = df["UnitsSold"]        # Dependent variable

model = LinearRegression()
model.fit(X, y)

intercept   = model.intercept_
coefficient = model.coef_[0]
r2          = r2_score(y, model.predict(X))

print("=== Regression Results ===")
print(f"Intercept        : {intercept:.2f}")
print(f"Coefficient      : {coefficient:.2f}")
print(f"R² Score         : {r2:.4f}")
print()
print(f"Equation  →  UnitsSold = {intercept:.2f} + {coefficient:.2f} × FootTraffic")
print()

# ------------------------------------------------------------
# 3. PREDICTIONS
# ------------------------------------------------------------
df["Predicted_UnitsSold"] = model.predict(X).round(0).astype(int)
print("=== Actual vs Predicted ===")
print(df[["Month", "FootTraffic", "UnitsSold", "Predicted_UnitsSold"]].to_string(index=False))
print()

# ------------------------------------------------------------
# 4. VISUALISATION — Regression Plot
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["FootTraffic"], df["UnitsSold"],
            color="#2563EB", label="Actual Sales", zorder=5, s=80)

x_line = np.linspace(df["FootTraffic"].min(), df["FootTraffic"].max(), 200)
y_line = intercept + coefficient * x_line
plt.plot(x_line, y_line, color="#DC2626", linewidth=2, label="Regression Line")

plt.xlabel("Foot Traffic (customers/month)", fontsize=12)
plt.ylabel("Units Sold", fontsize=12)
plt.title("Smart VendAI — Foot Traffic vs Units Sold\nSimple Linear Regression", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("regression_plot.png", dpi=150)
plt.show()
print("Plot saved as regression_plot.png")

# ------------------------------------------------------------
# 5. INTERPRETATION
# ------------------------------------------------------------
print()
print("=== Interpretation ===")
print(f"• A coefficient of {coefficient:.2f} means that for every additional")
print(f"  customer passing by, approximately {coefficient:.2f} extra units are sold.")
print(f"• R² = {r2:.4f} → the model explains {r2*100:.1f}% of the variance in sales.")
print("• High foot traffic locations are strongly recommended for Smart VendAI deployment.")
