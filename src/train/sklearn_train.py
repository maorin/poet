# -*- coding: utf-8 -*-

import numpy as np
import json, codecs
from sklearn.externals import joblib

def loadDataSet():
    '''
    把商品名称经过jieba分词为list结构 保存为文件new_data
    title_list  为商品标题分词后的list
    name 商品分类名称
    '''
    postingList = []
    classVec = []
    f = codecs.open('ready_data/new_data', 'rb',encoding='utf-8')
    data = json.load(f)
    
    for a in data:
        postingList.append(a['title_list'])
        classVec.append(a['name'])
    return postingList,classVec
#准备好数据  listOPosts为标题分词后的list, listClasses 每一个商品对应的分类
listOPosts, listClasses = loadDataSet()
print listOPosts[0]

#把测试数据转换成矩阵
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(listOPosts)
y = listClasses
print dir(X)
print X[0]
print len(X[0:1][0])
print X
print len(X)
print len(y)
print len(listClasses)

# mlb.classes_为训练时不重复的词的数组 预测时需要把目标词组转换成这个模型对应的数组

print mlb.classes_


#fit模型填充  predict训练  
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
y_predicted = clf.predict(X)

#模型评估
print np.mean(y_predicted == y)
#结果为 0.764768129474
from sklearn import metrics
print(metrics.classification_report(y, y_predicted))
print(metrics.confusion_matrix(y, y_predicted))



#保存结果
joblib.dump(clf, 'data/jueju_clf.pkl') 
joblib.dump(mlb, 'data/jueju_mlb.pkl')

#读结果
clf = joblib.load('data/jueju_clf.pkl') 
