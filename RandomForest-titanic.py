# -*- coding: utf-8 -*-
# @Time    : 2019/2/17 10:46
# @Author  : jiyl
# @File    : RandomForest-titanic.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer

"""
随机森林对泰坦尼克号进行预测生死
"""
# 获取数据

titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

x = titan[['pclass', 'age', 'sex']]
y = titan['survived']

x1 = x.fillna(x.mean()['age']).copy()

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.25)

# 进行处理（特征工程）特征-》类别-》one_hot编码
dict = DictVectorizer(sparse=False)
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
print(dict.get_feature_names())
x_test = dict.transform(x_test.to_dict(orient="records"))

rf = RandomForestClassifier(n_jobs=-1)

param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

# 网格搜索与交叉验证
gc = GridSearchCV(rf, param_grid=param, cv=2)

gc.fit(x_train, y_train)

score = gc.score(x_test, y_test)

print(score)
print("查看选择的参数模型：", gc.best_params_)
