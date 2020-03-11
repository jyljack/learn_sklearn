from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
