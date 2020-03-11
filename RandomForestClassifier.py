# -*- ecoding: utf-8 -*-
# @Author: JylJack
# @Time: 2020/3/11 14:18
# @FileName: RandomForestClassifier
# 随机森林分类

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(X_train, y_train)

clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)

score_c = clf.score(X_test, y_test)
score_r = rfc.score(X_test, y_test)

print("Single Tree:{}".format(score_c), "Random Forest:{}".format(score_r))

rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)
clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)
plt.plot(range(1, 11), rfc_s, label="RandomForest")
plt.plot(range(1, 11), clf_s, label="Decision Tree")
plt.legend()
plt.show()
