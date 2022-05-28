# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE

dataset = pd.read_csv('ParisHousing.csv')

# train/test split
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# isolate our target variable
label_train = train_set["price"].copy()
labels_test = test_set["price"].copy()
train_set = train_set.drop(['price'], axis=1)
test_set = test_set.drop(['price'], axis=1)

# Define the Linear Regression that we will use
lin_reg = LinearRegression()

# We have 16 features; now let's go through all of them and see what number of features offers the best, lowest RMSE
# on the training set:

# testing RFE on TRAIN
for i in range(1, 17):
    rfe_i = RFE(lin_reg, step=i)
    rfe_i = rfe_i.fit(train_set, label_train)
    predictions_rfe_i = rfe_i.predict(train_set)
    lin_mse_rfe_i = mean_squared_error(label_train, predictions_rfe_i)
    lin_rmse_rfe_i = np.sqrt(lin_mse_rfe_i)
    print(i, " ", lin_rmse_rfe_i)

# So we see that keeping between 1 and 3 features gives a huge error, so we cannot keep only 3 of them. However,
# between 8 and 16 the error is almost constant, and around 1890. Let's check the target variable, to see what this
# means compared to it:
rounded = round(dataset['price'].describe())
print(rounded)

# Now let's keep only 8 features and apply it on the train set, and then try it on the test set:

# train set
rfe = RFE(lin_reg, step=8)
rfe = rfe.fit(train_set, label_train)
predictions_rfe = rfe.predict(train_set)
lin_mse_rfe = mean_squared_error(label_train, predictions_rfe)
lin_rmse_rfe = np.sqrt(lin_mse_rfe)
lin_rmse_rfe

# Saving model to disk
pickle.dump(rfe, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))

# test set
predictions_test_rfe = model.predict(test_set)
lin_mse_test_rfe = mean_squared_error(labels_test, predictions_test_rfe)
lin_rmse_test_rfe = np.sqrt(lin_mse_test_rfe)

print(lin_rmse_test_rfe)
# So, I managed to reduce the number of features from 16 to 8 and have an error close to the one corresponding to a
# regression with all 16 features.
