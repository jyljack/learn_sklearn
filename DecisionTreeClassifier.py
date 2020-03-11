# -*- ecoding: utf-8 -*-
# @Author: JylJack
# @Time: 2020/3/11 14:02
# @FileName: DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(score)
