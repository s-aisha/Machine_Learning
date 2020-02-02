


from sklearn.model_selection import  train_test_split
from sklearn.datasets import  load_iris
from sklearn import load_tree

data  = load_tree()

# print(data)

x = data.data
y = data.target


xtrain, xtest, ytrain, ytest =  train_test_split(x,y,random_state=0)


from sklearn.tree import DecisionTreeClassifier, tree

clf  = DecisionTreeClassifier()

clf.fit(xtrain,ytrain)



ypred_train = clf.predict(xtrain)


from sklearn.tree import  export_graphviz
import pydotplus


dot_data=export_graphviz(clf,out_file=None,feature_names=data.feature_names,class_names=data.target_names)
print(dot_data)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree.pdf")












