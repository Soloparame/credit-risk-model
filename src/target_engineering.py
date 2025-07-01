import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def calculate_rfm(df, customer_id_col, transaction_date_col, amount_col, snapshot_date=None):
    df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])

    if snapshot_date is None:
        snapshot_date = df[transaction_date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id_col).agg({
        transaction_date_col: lambda x: (snapshot_date - x.max()).days,
        customer_id_col: 'count',
        amount_col: 'sum'
    }).rename(columns={
        transaction_date_col: 'recency',
        customer_id_col: 'frequency',
        amount_col: 'monetary'
    }).reset_index()

    return rfm
def create_rfm_labels(rfm_df, n_clusters=3, random_state=42):
    # Normalize RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Analyze clusters
    cluster_stats = rfm_df.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()

    # High-risk = cluster with lowest frequency & monetary (possibly high recency too)
    high_risk_cluster = cluster_stats.sort_values(by=['frequency', 'monetary', 'recency']).index[0]

    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)

    return rfm_df[[rfm_df.columns[0], 'is_high_risk']]
def assign_proxy_target(df_raw, df_processed, customer_id_col='customer_id', transaction_date_col='transaction_time', amount_col='transaction_amount'):
    # 1. Calculate RFM metrics
    rfm_df = calculate_rfm(df_raw, customer_id_col, transaction_date_col, amount_col)

    # 2. Cluster and assign high-risk label
    rfm_labeled = create_rfm_labels(rfm_df)

    # 3. Merge into processed dataset
    df_labeled = df_processed.merge(rfm_labeled, on=customer_id_col, how='left')

    # Handle any missing labels (e.g., due to missing transaction history)
    df_labeled['is_high_risk'] = df_labeled['is_high_risk'].fillna(0).astype(int)

    return df_labeled
