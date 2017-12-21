import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def predict(x):
    return slope * x + intercept

#generate random values 
pageSpeed = np.random.normal(3.0, 5.0, 1000)
purchaseAmount = 100 - (pageSpeed + np.random.normal(0, 1.9, 1000)) * 3

#perform linear regression extracting the r_value
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeed, purchaseAmount)

#print how well linear regression fits this model using r2
print(r_value ** 2)

fitLine = predict(pageSpeed)

#plot and show our findings
plt.scatter(pageSpeed, purchaseAmount)
plt.plot(pageSpeed, fitLine, c='r')
plt.show()