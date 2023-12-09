#Author : Dr. Ponraj Arumugam
# Wageningen Environmental Research
# load required libraries here
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np
from sklearn import datasets, ensemble
from joblib import dump, load

train_df = pd.read_csv("C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data_to_process\\sheets\\df_all_filled_2006_2016_new.csv")
df1 = train_df.dropna()
values_to_drop = [2015,2016]
values_to_drop1 = [2015,2016]
df = df1[~df1['Year'].isin(values_to_drop1)]
df2 = df1[df1['Year'].isin(values_to_drop)]

#data_X = df.drop(["Year","Yield"],axis=1)
data_X = df.drop(["Yield"],axis=1)
data_y = df['Yield']
#df_all = train_df.drop(["Lat","Lon","lat_lon"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)
#X_test1 = df2.drop(["Year","Yield"],axis=1)
X_test1 = df2.drop(["Yield"],axis=1)
y_test1 = df2['Yield']

new = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                learning_rate=0.07, loss='squared_error', max_depth=25,
                                max_features='sqrt', max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_samples_leaf=200, min_samples_split=20,
                                min_weight_fraction_leaf=0.0, n_estimators=300,
                                n_iter_no_change=None,
                                random_state=0, subsample=1, tol=0.0001,
                                validation_fraction=0.1, verbose=0, warm_start=False)

new.fit(X_train, y_train)
y_pred = new.predict(X_test1)
mse = mean_squared_error(y_test1, y_pred)
print(f'The mean squared error (MSE) on test set: {mse:.4f}')
r2 = r2_score(y_test1, y_pred)
# Print the R-squared value
print(f'The R2 on test set: {r2:.4f}')
model_filename = 'C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\models\\GBR.joblib'
dump(new, model_filename)
# Later, when you want to make predictions with the saved model:
# Load the model
#loaded_model = load(model_filename)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#plt.plot(X, y, c='k', label='data')
# Create a scatter plot
axs[0].scatter(y_test1, y_pred)
# Add a 1:1 line
axs[0].plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)], linestyle='--', color='gray', label='1:1 line')
# Perform linear regression to calculate R2
# Add labels and a title
axs[0].set_xlabel('Observed')
axs[0].set_ylabel('Predicted')
axs[0].set_title('Observed Vs Predicted')
axs[0].annotate(f'R2 = {r2:.2f}', xy=(0.1, 0.85), xycoords='axes fraction', fontsize=18)
axs[0].annotate(f'MSE = {mse:.2f}', xy=(0.1, 0.75), xycoords='axes fraction', fontsize=18)
# Show the plot

feature_importance = new.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
top_n = 10
top_feature_indices = sorted_idx[-top_n:]
top_feature_importance = feature_importance[top_feature_indices]
top_feature_names = np.array(X_train.columns)[top_feature_indices]
# Create the plot
axs[1].barh(pos[-top_n:], top_feature_importance, align="center")
axs[1].set_yticks(pos[-top_n:], top_feature_names)
axs[1].set_title("Top 10 Feature Importance")
axs[1].set_xlabel("Importance")
# Show the plot
plt.show()

plt.savefig("C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data_to_process\\maps\\first_results1.png",dpi=500)

