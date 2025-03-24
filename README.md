# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select three features as X and two target variables as Y, then split into train and test sets.

2.Standardize X and Y using StandardScaler for consistent scaling across features.

3.Initialize SGDRegressor and wrap it with MultiOutputRegressor to handle multiple targets.

4.Train the model on the standardized training data.

5.Predict on the test data, inverse-transform predictions, compute mean squared error, and print results.


## Program:
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: TRISHA PRIYADARSHNI PARIDA
RegisterNumber:  212224230293
*/

```
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

df.info()


X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()

Y=df[['AveOccup','HousingPrice']]
Y.info()


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print(Y_pred)
```








## Output:

![Screenshot 2025-03-24 105515](https://github.com/user-attachments/assets/9771f828-a2ed-4e90-a160-11a7dc571df2)

![Screenshot 2025-03-24 105522](https://github.com/user-attachments/assets/2ae0f6bc-07ec-495d-ba2b-3e697786ccab)

![Screenshot 2025-03-24 105529](https://github.com/user-attachments/assets/1d26f7de-0e69-48c7-8a0e-7f8d30bab9e5)

![Screenshot 2025-03-24 105534](https://github.com/user-attachments/assets/562ddc17-8a50-4e3a-8030-5e66641d96ce)

![Screenshot 2025-03-24 105545](https://github.com/user-attachments/assets/6ed410ca-c446-438f-9d18-a30200afb045)

![Screenshot 2025-03-24 105553](https://github.com/user-attachments/assets/224c54a5-c57f-44f4-a435-289342fb6a57)

![Screenshot 2025-03-24 105604](https://github.com/user-attachments/assets/4171c2bd-fd43-44ce-b36f-cf250ff58d5e)

![Screenshot 2025-03-24 105618](https://github.com/user-attachments/assets/8c7af457-1f6b-4536-93f2-d0597057a1b7)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
