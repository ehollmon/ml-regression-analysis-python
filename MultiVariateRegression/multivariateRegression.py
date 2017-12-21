import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

#import our data set
df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

scale = StandardScaler()

#colums used to predict
x = df[['Mileage', 'Cylinder', 'Doors']]

#value I'm trying to predict
y = df['Price']

# generate coefficients
x[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(x[['Mileage', 'Cylinder', 'Doors']].as_matrix())

print(x)

# create and display OLS model
est = sm.OLS(y,x).fit()
print(est.summary())