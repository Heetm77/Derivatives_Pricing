# Derivatives_Pricing

A quantitative finance project focused on **derivatives pricing and risk modeling**, implemented from first principles using analytical, numerical, and Monte Carlo methods.

This repository is designed as a **portfolio-grade quant project**, emphasizing correctness, validation, and variance reduction techniques used in real trading and risk systems.

## ⭐ Highlights
- American option pricing via Longstaff–Schwartz Monte Carlo
- Early-exercise boundary estimation and visualization
- Regression overfitting diagnostics using path-split LSM
- Benchmarked against binomial trees

---

## 📌 Models Implemented

### Black–Scholes (Analytical)
- European call and put pricing
- Closed-form Greeks:
  - Delta (call & put)
  - Gamma
  - Vega
  - Theta

---

### Numerical Greeks (Finite Differences)
- Delta, Gamma, Vega, Theta
- Central-difference schemes
- Validated against analytical Black–Scholes Greeks

---

### Monte Carlo Pricing (GBM)
- Risk-neutral GBM simulation
- European call and put pricing
- Convergence validated against Black–Scholes prices

---

## 🇺🇸 American Option Pricing (Longstaff–Schwartz)

- Implemented American put pricing using the **Longstaff–Schwartz Monte Carlo (LSM)** algorithm
- Estimated continuation values via polynomial regression and analyzed **basis-degree sensitivity**
- Extracted and visualized the **optimal early-exercise boundary** for American puts
- Diagnosed **regression overfitting** by comparing in-sample LSM against **path-split (out-of-sample) LSM**
- Demonstrated stabilization of option prices under path-splitting and validated results against a **binomial tree benchmark**

---

## 📐 Monte Carlo Greeks

### Delta
- **Bump-and-Revalue (Finite Difference)**  
- **Pathwise Estimator (Low Variance)**

### Vega
- **Likelihood Ratio Method (LRM)**  
- Correctly accounts for volatility dependence in the probability measure

All Monte Carlo Greeks are validated against analytical benchmarks.

---

## 🚀 Variance Reduction Techniques

### Control Variates
- Black–Scholes–based control variate
- Uses discounted terminal stock price with known expectation
- Achieves **~10× variance reduction** compared to plain Monte Carlo

---

## 📊 Key Results
- Monte Carlo prices converge to analytical values
- Pathwise Delta converges faster than bump-and-revalue
- Control variates significantly reduce estimator variance
- Path-splitting reveals upward bias in high-degree LSM regressions due to overfitting

---

## 🛠️ Tech Stack
- Python
- NumPy, SciPy
- Object-oriented, modular design
- Git-based development workflow

---

## 🔮 Planned Extensions
- Implied volatility calibration
- Path-dependent options (Asian options)
- Stochastic volatility models (Heston)

---

## 📎 Disclaimer
This project is for **educational and portfolio purposes only** and does not constitute financial advice.