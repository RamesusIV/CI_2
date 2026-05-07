import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib as jb

x=np.random.rand(100,1)
y=2*x+3
model=LinearRegression()
model.fit(x,y)
jb.dump(model, "src/model.pkl")
print("Done bro")