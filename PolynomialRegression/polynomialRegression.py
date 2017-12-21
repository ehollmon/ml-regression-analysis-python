import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#generate a polynomial data set
np.random.seed(2)
pageSpeed = np.random.normal(3.0, 1.0, 1000)
purchaseAmount =  np.random.normal(50.0, 10.0, 1000) / pageSpeed

x = np.array(pageSpeed)
y = np.array(purchaseAmount)

#specify that we want a 4th degree polynomial fit to this data
p4 = np.poly1d(np.polyfit(x,y,4))

#display on a scatterplot graph
xp = np.linspace(0,7,100)
plt.scatter(x,y)
plt.plot(xp, p4(xp), c='r')
plt.show()

#calculate and measure r^2 errors 
r2 = r2_score(y, p4(x))
print(r2)

