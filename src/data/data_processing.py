import os

import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT, 'data')

# https://www.zillow.com/research/data/
df = pd.read_csv(os.path.join(DATA_PATH, 'raw', 'data.csv'))

# Drop unecessary columns
df = df.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])

# Mean Impute missing values
imputer = SimpleImputer(strategy='mean')
df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

# One-Hot Encode
df = pd.get_dummies(df, columns=['RegionName'])

# Save
df.to_csv(os.path.join(DATA_PATH, 'processed', 'processed.csv'), index=False)

