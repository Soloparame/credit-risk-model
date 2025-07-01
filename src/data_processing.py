import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# WoE & IV Encoding
from woe import WoE # type: ignore
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])

        df["transaction_hour"] = df[self.datetime_column].dt.hour
        df["transaction_day"] = df[self.datetime_column].dt.day
        df["transaction_month"] = df[self.datetime_column].dt.month
        df["transaction_year"] = df[self.datetime_column].dt.year

        return df.drop(columns=[self.datetime_column])

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_col='customer_id', agg_col='transaction_amount'):
        self.groupby_col = groupby_col
        self.agg_col = agg_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        grouped = X.groupby(self.groupby_col)[self.agg_col].agg([
            'sum', 'mean', 'count', 'std'
        ]).reset_index()

        grouped.columns = [self.groupby_col,
                           'total_transaction_amount',
                           'avg_transaction_amount',
                           'transaction_count',
                           'std_transaction_amount']

        X = pd.merge(X, grouped, on=self.groupby_col, how='left')
        return X
def build_pipeline(categorical_features, numerical_features, datetime_col=None):
    steps = []

    if datetime_col:
        steps.append(("extract_datetime", DateTimeFeatureExtractor(datetime_col)))

    steps.append(("aggregate", AggregateFeatures()))

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    steps.append(("preprocessor", preprocessor))

    pipeline = Pipeline(steps)
    return pipeline
def apply_woe(df, target_col, categorical_features):
    woe_encoder = WoE()
    woe_encoder.fit(df[categorical_features], df[target_col])
    woe_encoded = woe_encoder.transform(df[categorical_features])

    df_woe = df.drop(columns=categorical_features).join(woe_encoded)
    return df_woe
def preprocess_data(df, target_col='default', datetime_col='transaction_time'):
    categorical = ['gender', 'account_type']  # Example
    numerical = ['transaction_amount']        # Add more if available

    df = apply_woe(df, target_col, categorical)
    
    pipeline = build_pipeline(
        categorical_features=[],  # After WoE no need for One-Hot
        numerical_features=numerical + [
            'total_transaction_amount', 'avg_transaction_amount',
            'transaction_count', 'std_transaction_amount',
            'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year'
        ],
        datetime_col=datetime_col
    )

    features = pipeline.fit_transform(df)
    labels = df[target_col]
    return features, labels

