# -*- ecoding: utf-8 -*-
# @Author: JylJack
# @Time: 2020/3/11 14:18
# @FileName: KNeighborsClassifier
# KNN 近邻算法

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()

x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target, test_size=0.3)
knn_classifier=KNeighborsClassifier(6)
knn_classifier.fit(x_train,y_train)
y_predict=knn_classifier.predict(x_test)
score=knn_classifier.score(x_test,y_test)
print(score)