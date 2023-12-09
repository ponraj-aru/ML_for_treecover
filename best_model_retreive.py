#Author : Dr. Ponraj Arumugam
# Wageningen Environmental Research

# load required libraries here
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

train_df = pd.read_csv("C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data_to_process\\sheets\\df_all_filled_2006_2016_new.csv")
df1 = train_df.dropna()
values_to_drop = [2015,2016]
values_to_drop1 = [2009,2015,2016]
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
cols = ['learning_rate', 'n_estimators', 'max_depth','min_sample_split','min_sample_leaf','sub_sample',"r2_test","r2_train","r2_all","r2_test1","mse_test","mse_train","mse_all","mse_test1","k_fold_3"]
lst = []
#lr = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009] #learning rate
lr = [0.09,0.15,0.1,0.09,0.06,0.05,0.01,0.009,0.005,0.001] #learning rate
#ne = [1750,2000,2500,3000,4000,5000,6000,7000] #n_estimator
ne = [200,200,300,400,500,600,700,800,900,1000,3000] #n_estimator
md = [13,9,10,11,12,13] #max_depth
mss = [300,4,6,8,10,20,40,60,100] #min_sample_split
msl = [50,3,5,7,9,15,20,25,30] #min_sample_leaf
ss = [1,0.85,0.9,0.95,1] #sub_sample
sr_n=1
for i in lr:
    for j in ne:
        for k in md:
            for l in mss:
                for m in msl:
                    for n in ss:
                        new = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                                        learning_rate=i, loss='squared_error', max_depth=k,
                                                        max_features='sqrt', max_leaf_nodes=None,
                                                        min_impurity_decrease=0.0,
                                                        min_samples_leaf=m, min_samples_split=l,
                                                        min_weight_fraction_leaf=0.0, n_estimators=j,
                                                        n_iter_no_change=None,
                                                        random_state=10, subsample=n, tol=0.0001,
                                                        validation_fraction=0.1, verbose=0, warm_start=False)
                        print(str(i),"--",str(j),"--",str(k),"--",str(l),"--",str(m),"--",str(n))
                        new.fit(X_train, y_train)
                        r2_test = r2_score(y_test, new.predict(X_test))
                        #r2_test = format(new.score(X_test, y_test))
                        r2_train = r2_score(y_train, new.predict(X_train))
                        r2_all = r2_score(data_y, new.predict(data_X))
                        r2_test1 = r2_score(y_test1, new.predict(X_test1))
                        mse_test = mean_squared_error(y_test, new.predict(X_test))
                        mse_train = mean_squared_error(y_train, new.predict(X_train))
                        mse_all = mean_squared_error(data_y, new.predict(data_X))
                        mse_test1 = mean_squared_error(y_test1, new.predict(X_test1))
                        r2_scores = cross_val_score(new, X_train, y_train, cv=3, scoring='r2')
                        # Calculate the mean R-squared score
                        k_fold = r2_scores.mean()
                        lst.append([i, j, k, l, m, n, r2_test, r2_train, r2_all,r2_test1, mse_test, mse_train, mse_all,mse_test1,k_fold])
                        sr_num = sr_n + 1
                        #print(str(sr_num), '-', 'lr:', str(i), 'ns:', str(j), 'md:', str(k))
                        print('Accuracy of the GBM on test set: {:.3f}'.format(r2_test))
                        print('Accuracy of the GBM on train set: {:.3f}'.format(r2_train))
                        print('Accuracy of the GBM on whole set: {:.3f}'.format(r2_all))
                        print('Accuracy of the GBM on out of sample data: {:.3f}'.format(r2_test1))
                        print('Accuracy of the GBM on k-fold 3: {:.3f}'.format(k_fold))
                        mse1 = mean_squared_error(y_test, new.predict(X_test))
                        print("MSE of test of data: %.4f" % mse1)
                        mse2 = mean_squared_error(y_train, new.predict(X_train))
                        print("MSE of train of data: %.4f" % mse2)
                        mse3 = mean_squared_error(data_y, new.predict(data_X))
                        print("MSE of whole of data: %.4f" % mse3)
                        df1 = pd.DataFrame(lst, columns=cols)
                        df1.to_csv(str(sr_num)+'.csv')

df1 = pd.DataFrame(lst, columns=cols)
df1.to_csv('hyper_parameter_all.csv')