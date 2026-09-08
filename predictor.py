import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df=pd.read_csv("Salary_dataset.csv")

print(df.head())
X=df.drop("Salary", axis=1)
y=df["Salary"].values

X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

model=LinearRegression()

model.fit(X_train, y_train)

print("Coefficient:",model.coef_)
print("Intercept:", model.intercept_)

y_pred=model.predict(X_test)
print(y_pred)
print("Actual:", y_test)
print("Predicted:", y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)


plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Slary")
plt.ylabel("Predicted Salary")
plt.title("Predicted vs Actual Salary")
plt.show()

