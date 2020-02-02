
import pandas as pd


df = pd.read_csv('rainfall.csv',delimiter=',')

columns = df.columns

x = df[columns[2:len(columns)-1]].values
y = df[columns[len(columns)-1]].values



from sklearn import  model_selection
from  sklearn.linear_model import LinearRegression

xtrain , xtest , ytrain , ytest = model_selection.train_test_split(x,y,random_state=0)

alg = LinearRegression()

alg.fit(xtrain,ytrain)

ypred = alg.predict(xtest)

# find the score of the algorithm

score = alg.score(xtest,ypred)
print('score : ', score)


print('M value : ',alg.coef_)
print('C value : ',alg.intercept_)


import matplotlib.pyplot as plt
import  numpy as np

m = alg.coef_[0]
c = alg.intercept_

x = np.arange(1,100,.1)

y = m*x + c

# plt.plot(x,y, color ='g',label = 'best line found by algo to predict future',linewidth = 9)


plt.plot(xtrain,ytrain, color ='r',label = 'linear function learnt by algo')

plt.title('x == y')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()

plt.show()


import matplotlib.pyplot as plt
plt.plot(xtest,ypred, color ='g',label = 'Prediction',linewidth = 9)


plt.plot(xtest,ytest, color ='r',label = 'Test')

plt.title('x == y')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()

plt.show()




