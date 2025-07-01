
# Credit Risk Model

A credit scoring system that leverages data-driven approaches to assess the risk of loan default using machine learning. This repository includes code for data processing, model training, and a FastAPI-based API for inference.

---

## 📁 Project Structure

```bash
credit-risk-model/
├── .github/workflows/ci.yml         # CI/CD with GitHub Actions
├── data/                            # Ignored: contains raw and processed data
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb                # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py           # Feature engineering
│   ├── train.py                     # Model training logic
│   ├── predict.py                   # Prediction logic
│   └── api/
│       ├── main.py                  # FastAPI application
│       └── pydantic_models.py       # Pydantic schemas
├── tests/
│   └── test_data_processing.py      # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
📊 Credit Scoring Business Understanding
1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
Basel II requires banks to measure, report, and manage credit risk more rigorously. Financial institutions must calculate regulatory capital using Internal Ratings-Based (IRB) approaches. This emphasizes the need for transparency and accountability. Therefore, we must use interpretable models where decision logic can be explained to both internal risk committees and external regulators. Documentation, feature explanations (e.g., WoE), and audit trails are not optional—they’re critical for compliance.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Without explicit default labels (i.e., whether a borrower has failed to pay), we must create a proxy — for example, considering "30+ days past due" as a default event. However, proxies introduce ambiguity. A customer may delay payment for technical or personal reasons without being a true credit risk. The risk is that we may incorrectly classify a good customer as risky (False Positive) or worse, approve a high-risk borrower (False Negative), leading to loan loss.

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Simple Models (e.g., Logistic Regression + WoE):

✅ Easy to explain and audit

✅ Fast training, works well with small data

❌ May underperform on complex patterns

Complex Models (e.g., Gradient Boosting Machines):

✅ Higher predictive power

❌ Poor interpretability

❌ Harder to justify decisions to regulators

In regulated contexts, especially under Basel II/III, interpretable models are often favored, unless complex models are paired with explainability techniques like SHAP or LIME, and rigorous validation and fairness audits are performed.

📌 Future Plans
Add SHAP explainability for model decisions

Compare model families (Logistic, Random Forest, XGBoost)

Deploy REST API with FastAPI

Add model versioning with MLflow or DVC

📚 References
Statistica Sinica: Credit Risk Modeling

Alternative Credit Scoring (HKMA)

World Bank Credit Scoring Guide

How to Build a Credit Risk Scorecard

Corporate Finance Institute

Risk Officer: Credit Risk