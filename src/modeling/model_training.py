import joblib
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'processed')
MODELS_PATH = os.path.join(ROOT, 'models')

df = pd.read_csv(os.path.join(DATA_PATH, 'processed.csv'))

x = df.drop(columns=['2025-01-31'])
y = df['2025-01-31']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

predicted = model.predict(x_test)

print(f"MSE: {mean_squared_error(y_test, predicted): ,.2f}")
print(f"R2 Score: {r2_score(y_test, predicted): .4f} ")

joblib.dump(model, os.path.join(MODELS_PATH, 'random_forest.pkl'))
