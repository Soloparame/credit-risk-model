from src.data_processing import preprocess_data
import pandas as pd

df = pd.read_csv("data/raw/credit_data.csv")
X, y = preprocess_data(df, target_col='default', datetime_col='transaction_time')
