# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.

2.Load the dataset into a Pandas DataFrame and preview it using head() and tail().

3.Extract the independent variable X and dependent variable Y from the dataset.

4.Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.

5.In a loop over the number of epochs:

.Compute the predicted value Y_pred using the formula . Calculate the gradients

.Update the parameters m and c using the gradients and learning rate.

.Track and store the error in each epoch.

6.Plot the error against the number of epochs to visualize the convergence.

7.Display the final values of m and c, and the error plot.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KEERTHANA S
RegisterNumber:  212223040092
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output:

![image](https://github.com/user-attachments/assets/45dec6fc-3df2-4a6b-882c-c4adba501d47)

```
dataset.info()

```
##Output:

![image](https://github.com/user-attachments/assets/e21fb695-1d16-4ffa-ae8d-53f90f924eae)

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)

```

##Output:


![image](https://github.com/user-attachments/assets/bc688dba-f8cb-4d7b-9649-9cb9af6d5d75)

```
print(X.shape)
print(Y.shape)

```
##Output:

![image](https://github.com/user-attachments/assets/ca480048-eb5f-400e-b461-fd4ad32b333d)


```
m=0
c=0
L=0.0001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
    Y_pred = m*X +c
    D_m=(-2/n)*sum(X *(Y-Y_pred))
    D_c=(-2/n)*sum(Y -Y_pred)
    m=m-L*D_m
    c=c-L*D_c
    error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
```

##Output:

![image](https://github.com/user-attachments/assets/820b5655-dd2c-4059-8385-aad6ae650310)

```
plt.plot(range(0,epochs),error)
```

##Output:

![WhatsApp Image 2024-09-03 at 18 32 43_2ec4d517](https://github.com/user-attachments/assets/435035b4-c01d-435c-930c-77f6991d4179)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
