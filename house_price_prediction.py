import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# https://www.zillow.com/research/data/
df = pd.read_csv('Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')

print(df.isna().sum().sum())
print(df.duplicated().sum().sum())

# Dataset has 0 Duplicates, and has Nan's Time to Visualise it
sns.heatmap(df.isna(), cmap='viridis', cbar=False)
plt.show()

impute = SimpleImputer(strategy='mean')
impute_cat = SimpleImputer(strategy='most_frequent')
df.iloc[:, 5:] = impute.fit_transform(df.iloc[:, 5:])
df.iloc[:, :5] = impute_cat.fit_transform(df.iloc[:, :5])
print(df.isna().sum().sum())

df_encoded = pd.get_dummies(df.iloc[:, :5])
x = pd.concat([df_encoded, df.iloc[:, 5: -1]], axis=1)
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R2 Score: {r2_score(y_test, y_pred) * 100:.2f}')

plt.title('Actual Vs Predicted Price')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.scatter(y_test, y_pred, alpha=0.5)
plt.show()
