


from sklearn.model_selection import  train_test_split
from sklearn.datasets import  load_breast_cancer


data  = load_breast_cancer()

# print(data)

x = data.data
y = data.target


xtrain, xtest, ytrain, ytest =  train_test_split(x,y,random_state=0)


from  sklearn.tree import DecisionTreeClassifier

clf  = DecisionTreeClassifier()

clf.fit(xtrain,ytrain)


ypred = clf.predict(xtest)


from sklearn.tree import  export_graphviz
import pydotplus


dot_data=export_graphviz(clf,out_file=None,feature_names=data.feature_names,class_names=data.target_names)
# print(dot_data)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("cancer.pdf")


from  sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))



print(clf.score(xtest,ytest))




