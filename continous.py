from sklearn.datasets import  load_iris
data  = load_iris()
x = data.data
y = data.target



from sklearn import  model_selection
from sklearn.metrics import classification_report,confusion_matrix

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x,y,test_size=0.25,random_state=0)

# # inbuilt naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(xtrain,ytrain)
ypred = clf.predict(xtest)
print("Score for algorithm : ",clf.score(xtest,ypred))
print(classification_report(ytest,ypred))
print(confusion_matrix(ytest,ypred))



import matplotlib.pyplot as plt


plt.plot(xtrain,ytrain, color ='r',label = 'Trained data')

plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()

plt.show()


import matplotlib.pyplot as plt
plt.plot(xtest,ypred, color ='g',label = 'Predicted data',linewidth = 9)


plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()

plt.show()

plt.plot(xtest,ytest, color ='r',label = 'Test data')


plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()

plt.show()











