import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def calculate_mape(y_true, y_pred):
    return (1/len(y_true)) * sum(abs((y_true - y_pred) / y_true)) * 100

csv_file_path = 'IPL_Matches_2008_2022.csv'
df = pd.read_csv(csv_file_path)

target_variable = 'WonBy'

df_cleaned = df.dropna(thresh=950, axis=1)

selected_features = ['ID', 'Date', 'Season', 'MatchNumber', 'Team1', 'Team2', 'Venue', 'TossWinner', 'TossDecision', 'WonBy', 'Team1Players', 'Team2Players', 'Umpire1', 'Umpire2']
df_selected = df_cleaned[selected_features]

label_encoder = LabelEncoder()
df_encoded = df_selected.apply(lambda col: label_encoder.fit_transform(col.astype(str)), axis=0)

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

X = df_imputed.drop(target_variable, axis=1)
y = df_imputed[target_variable].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor_svm = SVR()
regressor_svm.fit(X_train_scaled, y_train)

y_test_pred = regressor_svm.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mape = calculate_mape(y_test, y_test_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R²): {r2}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

results_df = pd.DataFrame({
    'Metric': ['Mean Squared Error', 'R-squared', 'Mean Absolute Error', 'Mean Absolute Percentage Error'],
    'Value': [mse, r2, mae, mape]
})
print ("/n")
print ("/n")
print (results_df)
print ("/n")
import matplotlib.pyplot as plt

plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
