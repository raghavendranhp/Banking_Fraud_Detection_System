# Banking Transaction Intelligence: Insights & Observations

## Executive Summary
This report summarizes the findings from the Banking Transaction Intelligence and Fraud Detection system. We analyzed transaction patterns, engineered behavioral features, trained multiple machine learning models to detect fraud, and developed a customer risk scoring mechanism.

---

## 1. Data Analysis & Feature Engineering Summary

### Data Exploration
- **Transaction Amount Distribution**: Fraudulent transactions generally have a different distribution of amounts compared to legitimate ones. (e.g. unusually high amounts or a series of small test amounts).
- **Merchant Risk**: Certain merchant categories exhibit a significantly higher historical fraud rate.

### Feature Engineering approach
To enable the models to detect complex patterns, we engineered the following features:
- **`account_tx_frequency`**: Captures high-velocity transaction bursts (often a sign of account takeover).
- **`account_avg_spending` & `unusual_tx_amount_ratio`**: Allows the model to identify transactions that are anomalous *for that specific customer*, rather than just globally high amounts.
- **`location_deviation`**: Flags transactions occurring outside the customer's home city, a classic fraud indicator.
- **`merchant_risk_score`**: Assigns a risk weight to the transaction based on the historical fraud rate of that merchant category.

---

## 2. Fraud Detection Model Performance

We evaluated standard classifiers alongside an anomaly detection baseline. 

* **Logistic Regression**: Serves as a strong, interpretable baseline.
* **Random Forest**: Captures non-linear relationships well and is robust to outliers.
* **XGBoost**: Typically the strongest performer on tabular data, balancing speed and high accuracy.
* **Isolation Forest**: Useful for detecting completely novel fraud patterns (zero-day fraud) that supervised models haven't seen.

### Metric Details:
- **Accuracy**: Overall correct prediction rate.
- **Precision**: Of all transactions flagged as fraud, how many actually were? (Crucial to minimize false positives, which annoy customers).
- **Recall**: Of all actual fraudulent transactions, how many did we catch? (Crucial to minimize financial loss).
- **F1 Score**: The harmonic mean of Precision and Recall, providing a balanced measure.
- **ROC-AUC**: Measures the model's ability to discriminate between classes across different thresholds.

---

## 3. Customer Risk Scoring Methodology

Instead of just looking at individual transactions, we developed a holistic risk profile for each customer by aggregating their data:

- **Volume Risk**: Customers with unusually high total transaction volumes relative to the base.
- **Activity Risk**: Customers exhibiting highly frequent transaction bursts.
- **Merchant Risk Exposure**: Customers who frequently transact in high-risk merchant categories.

The derived composite `raw_risk_score` (0-100) is bucketed into actionable categories:
*   **High Risk** (>75): Require immediate review by the fraud team. Consider temporary account freezes or additional authentication steps (e.g., OTP).
*   **Medium Risk** (40-75): Add to a monitoring watchlist.
*   **Low Risk** (<40): Normal account behavior.

---

## 4. Key Insights & Recommendations

1.  **Top Fraud Patterns**: High-value transactions at atypical locations (high `location_deviation` + high `unusual_tx_amount_ratio`) are the strongest indicators of fraud.
2.  **Model Selection**: Ensemble methods (Random Forest / XGBoost) generally outperform linear models on this data due to the complex, non-linear interactions between features like merchant risk and individual spending habits.
3.  **Proactive Risk Management**: The new Customer Risk Score allows the bank to move from a reactive posture (catching fraud as it happens) to a proactive one (monitoring high-risk accounts before large losses occur).


