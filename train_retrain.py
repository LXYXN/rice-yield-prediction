# train_retrain.py
# Usage: python train_retrain.py --file data/yield_df.csv
import argparse
import os
import json
import pickle
from math import sqrt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def build_and_train(df, model_out_dir):
    # normalize column name
    if 'hg/ha_yield' in df.columns:
        df = df.rename(columns={'hg/ha_yield':'yield'})
    # filter rice rows if present
    if 'Item' in df.columns:
        df = df[df['Item'].astype(str).str.lower().str.contains('rice')].copy()
    # keep typical features if present
    possible = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','Area']
    features = [c for c in possible if c in df.columns]
    if 'yield' not in df.columns:
        raise SystemExit("No 'yield' column found in CSV after renaming. Check CSV.")
    X = df[features].copy()
    y = df['yield'].values

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # identify numeric / categorical
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    # preprocess
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', num_pipe, numeric_cols), ('cat', cat_pipe, categorical_cols)], remainder='drop')

    # pipelines
    lr_pipe = Pipeline([('pre', pre), ('lr', LinearRegression())])
    rf_pipe = Pipeline([('pre', pre), ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])

    print("Training Linear Regression...")
    lr_pipe.fit(X_train, y_train)
    print("Training Random Forest...")
    rf_pipe.fit(X_train, y_train)

    def eval_pipe(p):
        preds = p.predict(X_test)
        return {
            'rmse': float(sqrt(mean_squared_error(y_test, preds))),
            'mae': float(mean_absolute_error(y_test, preds)),
            'r2': float(r2_score(y_test, preds))
        }

    metrics = {'linear_regression': eval_pipe(lr_pipe), 'random_forest': eval_pipe(rf_pipe)}
    print("Metrics:", metrics)

    os.makedirs(model_out_dir, exist_ok=True)
    # save models where Flask app expects them
    with open(os.path.join(model_out_dir, 'random_forest_kaggle.pkl'), 'wb') as f:
        pickle.dump(rf_pipe, f)
    with open(os.path.join(model_out_dir, 'linear_regression_kaggle.pkl'), 'wb') as f:
        pickle.dump(lr_pipe, f)

    # save results
    with open(os.path.join(model_out_dir, 'results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved models and results to", model_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='data/yield_df.csv', help='path to CSV file')
    parser.add_argument('--out', default='models', help='output folder for models (relative)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"CSV not found at {args.file}. Place your CSV there or run with --file <path>")

    df = pd.read_csv(args.file)
    build_and_train(df, args.out)