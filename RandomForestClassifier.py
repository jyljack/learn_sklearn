# -*- encoding: utf-8 -*-
# @Author: JylJack
# @Time: 2020/3/11 14:18
# @FileName: RandomForestClassifier
# 随机森林分类

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(x_train, y_train)
score = rfc.score(x_test, y_test)
print(score)