import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "Yili(process)_month.xlsx"
original_data = pd.read_excel(file_path)

X = original_data[['Atm','Vap','TEMP', 'MAX', 'MIN', 'MAX','Rel_Hum']]
#X = original_data[['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MXSPD', 'MAX', 'MIN']]
y = original_data['PRCP']

y = pd.to_numeric(y, errors='coerce')

X = X.dropna()
y = y[X.index]

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

n_features = X.shape[1]
n_features_to_select = min(5, n_features)

rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)
X_new = rfe.fit_transform(X, y)

selected_features = X.columns[rfe.support_]
print("Selected features (RFE with Random Forest):", selected_features)

model.fit(X, y)

feature_importances = model.feature_importances_

importance_df_all = pd.DataFrame(list(zip(X.columns, feature_importances)), columns=['Feature', 'Importance'])
importance_df_all = importance_df_all.sort_values(by='Importance', ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'italic'

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_all, palette='viridis', hue='Feature')
plt.title('Feature Importance from Random Forest (All Features)',fontsize=20)

plt.xlabel('Importance', fontsize=18)
plt.ylabel('Feature', fontsize=18)
plt.show()

importance_df_rfe = pd.DataFrame(list(zip(selected_features, feature_importances[rfe.support_])), columns=['Feature', 'Importance'])
importance_df_rfe = importance_df_rfe.sort_values(by='Importance', ascending=False)

