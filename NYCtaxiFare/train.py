# load some default Python modules
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import seaborn as sns
plt.style.use('seaborn-whitegrid')


lr = LinearRegression()
# read data in pandas dataframe
print ("[*] Reading from train data file...")
data = pd.read_csv('train.csv', nrows = 5000000, parse_dates=["pickup_datetime"])
print("Importing DONE!")

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) 

def feature_eng(data):
    print("[*] Dropping rows containing nulls...")
    print('Old size: %d' % len(data))
    data = data.dropna()
    print('New size: %d' % len(data))
    print("[*] Done!")
    
    print ("[*] Dropping fare amount outliers...")
    data = data[data['fare_amount'].between(left = 2.5, right = 100)]
    print('New size: %d' % len(data))
    print("[*] Done!")
    
    print ("[*] Dropping suspicious passenger counts...")
    data = data.loc[data['passenger_count'] < 6]
    print('New size: %d' % len(data))
    print ("[*] Done!")
    
    print ("[*] Dropping points out of the city...")
    data = data.loc[data['pickup_latitude'].between(40, 42)]
    data = data.loc[data['pickup_longitude'].between(-75, -72)]
    data = data.loc[data['dropoff_latitude'].between(40, 42)]
    data = data.loc[data['dropoff_longitude'].between(-75, -72)]
    print('New size: %d' % len(data))
    print ("[*] Done!")
    
    print ("[*] Adding new column to dataframe with distance in km...")
    data['distance_miles'] = distance(data.pickup_latitude, data.pickup_longitude, data.dropoff_latitude, data.dropoff_longitude)
    print("[*] Done!")
    
    print ("[*] Finishing data set..")
    data['hour'] = data.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
    data['year'] = data.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
    # Absolute difference in latitude and longitude
    data['abs_lat_diff'] = (data['dropoff_latitude'] - data['pickup_latitude']).abs()
    data['abs_lon_diff'] = (data['dropoff_longitude'] - data['pickup_longitude']).abs()
    print ("[*] Done!")
    return data


def plot_prediction_analysis(y, y_pred, figsize=(10,4), title=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c='red')
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel('$\hat{y}$')
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title('rmse = {:.2f}, evs = {:.2f}'.format(rmse, evs))
    
    axs[1].hist(y-y_pred, bins=50)
    avg = np.mean(y-y_pred)
    std = np.std(y-y_pred)
    axs[1].set_xlabel('$y - \hat{y}$')
    axs[1].set_title('Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(avg, std))
    
    if title!='':
        fig.suptitle(title)


def feature_eng_test(data):
    print("[*] Dropping rows containing nulls...")
    print('Old size: %d' % len(data))
    data = data.dropna()
    print('New size: %d' % len(data))
    print("[*] Done!")
    
    print ("[*] Dropping suspicious passenger counts...")
    data = data.loc[data['passenger_count'] < 6]
    print('New size: %d' % len(data))
    print ("[*] Done!")
    
    print ("[*] Dropping points out of the city...")
    data = data.loc[data['pickup_latitude'].between(40, 42)]
    data = data.loc[data['pickup_longitude'].between(-75, -72)]
    data = data.loc[data['dropoff_latitude'].between(40, 42)]
    data = data.loc[data['dropoff_longitude'].between(-75, -72)]
    print('New size: %d' % len(data))
    print ("[*] Done!")
    
    print ("[*] Adding new column to dataframe with distance in km...")
    data['distance_miles'] = distance(data.pickup_latitude, data.pickup_longitude, data.dropoff_latitude, data.dropoff_longitude)
    print("[*] Done!")
    
    print ("[*] Finishing data set..")
    data['hour'] = data.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
    data['year'] = data.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
    # Absolute difference in latitude and longitude
    data['abs_lat_diff'] = (data['dropoff_latitude'] - data['pickup_latitude']).abs()
    data['abs_lon_diff'] = (data['dropoff_longitude'] - data['pickup_longitude']).abs()
    print ("[*] Done!")
    return data

test_data = feature_eng(data)


features = ['distance_miles', 'passenger_count','year', 'hour', 'abs_lat_diff', 'abs_lon_diff']
X = test_data[features].values
y = test_data['fare_amount'].values

# # create training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=9914)

print("[*] Training the model...")
lr.fit(X, y)
print("[*] Done!")

print("[*] Saving the model...")
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
print ("[*] Done!")

# model_lin = Pipeline((
#         ("standard_scaler", StandardScaler()),
#         ("lin_reg", LinearRegression()),
#     ))

# model_lin.fit(X_train, y_train)

# y_train_pred = model_lin.predict(X_train)
# plot_prediction_analysis(y_train, y_train_pred, title='Linear Model - Trainingset')

# y_test_pred = model_lin.predict(X_test)
# plot_prediction_analysis(y_test, y_test_pred, title='Linear Model - Testset')


# print ("[*] Reading from test data file...")
# final_data = pd.read_csv('test.csv', nrows = 9914, parse_dates=["pickup_datetime"])
# print ("[*] Test data preperation...")
# final_data = feature_eng_test(final_data)
# XTEST = final_data[features].values
# print ("[*] Done!")


# filename = './output/baseline_linear'

# y_pred_final = model_lin.predict(X_test)

# submission = pd.DataFrame({'key': test_data.key, 'fare_amount': y_pred_final}, columns = ['key', 'fare_amount'])
# submission.to_csv('submission.csv', index = False)

# y = test_data['fare_amount'].values,y = test_data['fare_amount'].values,