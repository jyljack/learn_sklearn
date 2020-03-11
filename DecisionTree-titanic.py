# -*- coding: utf-8 -*-
# @Time    : 2019/2/17 10:29
# @Author  : jiyl
# @File    : DecisionTree-titanic.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

"""
决策树对泰坦尼克号进行预测生死
"""
# 获取数据

titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

# 处理数据，找出特征值和目标值
x = titan[['pclass', 'age', 'sex']]
y = titan['survived']

# 缺失值处理
x['age'].fillna(x['age'].mean(), inplace=True)

# 分割数据集到训练集合测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 进行处理（特征工程）特征-》类别-》one_hot编码
dict = DictVectorizer(sparse=False)
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
x_test = dict.transform(x_test.to_dict(orient="records"))

dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)

score = dec.score(x_test, y_test)
print(score)