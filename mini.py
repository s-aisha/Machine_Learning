
import  numpy as np


total_data = 0

def fit(xtrain, ytrain):
    dictionary = {}
    class_values = set(ytrain)
    for current_class in class_values:
        dictionary[current_class] = {}
        current_class_rows = (ytrain==current_class)
        x_current = xtrain[current_class_rows]
        y_current = ytrain[current_class_rows]
        num_features = xtrain.shape[1]
        dictionary[current_class]['total_count'] = len(y_current)
        for j in range(0,num_features):
            dictionary[current_class][j]= {}
            all_possible_val_of_feature = set(xtrain[:,j])
            for current_value in all_possible_val_of_feature:
                dictionary[current_class][j][current_value] = (x_current[:,j]==current_value).sum()
    return dictionary


def probability(dictionary, x, current_class):
    output = np.log(dictionary[current_class]['total_count'])-np.log(total_data)
    num_features = len(dictionary[current_class].keys())-1
    for j in range(num_features):
        xj = x[j]
        count_current_class_with_value_xj = dictionary[current_class][j][xj]+1
        count_current_class = dictionary[current_class]['total_count'] + len(dictionary[current_class][j].keys())
        current_xj_probability = np.log(count_current_class_with_value_xj)-np.log(count_current_class)
        output = output + current_xj_probability
    return output


def predictSinglePoint(dictionary, x):
    classes = dictionary.keys()
    best_p = -1000
    best_class = -1
    first_run  = True
    for current_class in classes:
        p_current_class = probability(dictionary, x, current_class)
        if(first_run or p_current_class>best_p):
            best_p = p_current_class
            best_class = current_class
        first_run = False
    return  best_class


def predict(dictionary,xtest):
    ypred = []
    for x in xtest:
        xclass = predictSinglePoint(dictionary, x)
        ypred.append(xclass)
    return ypred


def makeLabel(column):
    secondLimit = column.mean()
    firstLimit = secondLimit * 0.5
    thirdLimit = secondLimit * 1.5
    for i in range(0,len(column)):
        if column[i] < firstLimit :
            column[i] = 0
        elif column[i] <secondLimit :
            column[i] = 1
        elif column[i] <thirdLimit:
            column[i] = 2
        else:
            column[i] = 3
    return  column


from sklearn.datasets import  load_iris
data  = load_iris()
x = data.data
y = data.target

# print(data)

# label on all columns
for i in range(0, x.shape[-1]):
    x[:,i] = makeLabel(x[:,i])


from sklearn import  model_selection


xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x,y,test_size=0.25,random_state=0)

total_data =len(ytrain)

dictionary = fit(xtrain,ytrain)
ypred = predict(dictionary,xtest)


from sklearn.metrics import  classification_report,confusion_matrix
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





