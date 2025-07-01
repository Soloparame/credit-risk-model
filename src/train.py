from src.target_engineering import assign_proxy_target
import pandas as pd

# Load raw and processed data
df_raw = pd.read_csv("../data/raw/data.csv")
df_processed = pd.read_csv("../data/processed/features.csv")

# Add the proxy target column
df_labeled = assign_proxy_target(
    df_raw,
    df_processed,
    customer_id_col='customer_id',
    transaction_date_col='transaction_time',
    amount_col='transaction_amount'
)

# Save the labeled dataset
df_labeled.to_csv("data/processed/features_with_target.csv", index=False)
