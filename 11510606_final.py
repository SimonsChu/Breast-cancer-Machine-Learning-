
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import re
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl 
from sklearn.preprocessing import Imputer
from numpy import random
import seaborn as sb
### Set path to the data set
dataset_path = "D:\cancer_proteomes_CPTAC_itraq.csv"
clinical_info = "D:\clinical_data_breast_cancer.csv"
pam50_proteins = "D:\PAM50_proteins.csv"
 
## 数据载入，处理表头
data = pd.read_csv(dataset_path,header=0,index_col=0)
clinical = pd.read_csv(clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample
pam50 = pd.read_csv(pam50_proteins,header=0)
 
## 去除symbol、name两列无，3列重复，3列健康人数据
data.drop(['gene_symbol','gene_name','AO-A12D.01TCGA','C8-A131.01TCGA','AO-A12B.01TCGA','263d3f-I.CPTAC','blcdb9-I.CPTAC','c4155b-C.CPTAC'],axis=1,inplace=True)
 
## Change the protein data sample names to a format matching the clinical data set
data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)
 
## 数据转置
data = data.transpose()


# # 数据预处理的实现

# In[16]:

#数据读取
data.head(20)


# In[17]:

#缺失值的填补（用中位数填充）
imputer = Imputer(missing_values='NaN', strategy='median', axis=1)
imputer = imputer.fit(data)
data = imputer.transform(data)
d=pd.DataFrame(data)
data=d
d.head(10)


# In[18]:


#数据Z-scale 标准化
from sklearn import preprocessing
data_scaled = preprocessing.scale(d)
s=pd.DataFrame(data_scaled)
s.head(20)


# In[19]:

#数据正则化
normalizer=preprocessing.normalize(s,norm='l2')
s=pd.DataFrame(normalizer)
s.head(10)


# In[20]:

#异常值删除，对Z-scale中大于和小于3的值
p = preprocessing.scale(s)
n1=[]
for i in p:
    if  (i>3).all()&(i<-3).all(): continue
    n1.append(i)
n1=pd.DataFrame(n1)

#对删除的异常值填补（用中位数填充）
imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
imputer = imputer.fit(n1)
data = imputer.transform(n1)
d=pd.DataFrame(n1)
d.head(10)


# # 对PAM50 mRNA进行聚类分析

# In[3]:

## 数据载入，处理表头
data = pd.read_csv(dataset_path,header=0,index_col=0)
clinical = pd.read_csv(clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample
pam50 = pd.read_csv(pam50_proteins,header=0)
 
## 去除symbol、name两列无，3列重复，3列健康人数据
data.drop(['gene_symbol','gene_name','AO-A12D.01TCGA','C8-A131.01TCGA','AO-A12B.01TCGA','263d3f-I.CPTAC','blcdb9-I.CPTAC','c4155b-C.CPTAC'],axis=1,inplace=True)
 
## Change the protein data sample names to a format matching the clinical data set
data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)
 
## 数据转置
data = data.transpose()
## 对第二个文件进行表头处理
clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data.index],:]

## 将clinical中数据与77_cancer_proteomes中数据进行整合，放在一起
merged = data.merge(clinical,left_index=True,right_index=True)
 
## 更改名称方便识别
processed = merged
 
## 将第二问中用不到的clinical中部分数据进行删除
processed_numerical = processed.loc[:,[x for x in processed.columns if bool(re.search("NP_|XP_",x)) == True]]

#将第三个文件中出现的PAM50蛋白质与第一个文件对应进行筛选，得到有用的43个基因
processed_numerical_p50 = processed_numerical.ix[:,processed_numerical.columns.isin(pam50['RefSeqProteinID'])]
processed1=pd.DataFrame(processed_numerical_p50)
processed1.head(10)


# In[6]:

#对第二问中要用到的数据进行预处理

#缺失值的填补（用中位数填充）
imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
imputer = imputer.fit(processed_numerical_p50)
data_imputer = imputer.transform(processed_numerical_p50)

#数据Z-scale 标准化
from sklearn import preprocessing
data_scaled = preprocessing.scale(data_imputer)

#数据正则化处理
normalizer=preprocessing.normalize(data_scaled,norm='l2')
data_norm=pd.DataFrame(normalizer)
data_norm.head(10)


# In[7]:


# ## 利用KMeans进行聚类计算
n_clusters = [2,3,4,5,6,7,8,10,20,76]
def compare_k_means(k_list,data_norm):
## 将处理过的数据分成不同类别
    for k in k_list:
        clusterer = KMeans(n_clusters=k, n_jobs=4)
        clusterer.fit(data_norm)
        ## The higher (up to 1) the better
        print("Silhouette Coefficient for k == %s: %s" % (
        k, round(metrics.silhouette_score(data_norm, clusterer.labels_), 4)))
        ## The higher (up to 1) the better
        print("Homogeneity score for k == %s: %s" % (
        k, round(metrics.homogeneity_score(processed['PAM50 mRNA'], clusterer.labels_),4)))
        print("------------------------")

## 从大的蛋白质库里随机取43个蛋白质
processed_numerical_random = processed_numerical.iloc[:,random.choice(range(processed_numerical.shape[1]),43)]
imputer_rnd = imputer.fit(processed_numerical_random)
processed_numerical_random = imputer_rnd.transform(processed_numerical_random)



## 利用筛选出的43蛋白质进行聚类分析
compare_k_means(n_clusters,data_norm)


## 将随机选取的43个蛋白质与筛选出的43个进行结果对照
compare_k_means(n_clusters,processed_numerical_random)
## The scores should be significantly lower than for the PAM50 proteins!


# In[8]:

clusterer_final = KMeans(n_clusters=3, n_jobs=4)
clusterer_final = clusterer_final.fit(data_norm)
processed_p50_plot = pd.DataFrame(data_norm)
processed_p50_plot['Gene'] = clusterer_final.labels_
processed_p50_plot.sort('Gene',axis=0,inplace=True)
 
## Look at the heatmap of protein expression in all patients and look at their assigned cluster
## Proteins can either be more expressed (more is produced, less degraded), not changed or lower expressed than the used reference
## Since each protein has a distinct function in the cell, their levels describe the functional/signaling state the cell is in.
#画出热力图
processed_p50_plot.index.name = 'Patient'
s=sb.heatmap(processed_p50_plot)
pl.show()


# # 问题三：对77个病人进行分类

# In[125]:

#训练集、测试集的划分，利用77个病人的43个蛋白质作为分类标准，
from sklearn import cross_validation
Y=processed['PAM50 mRNA']
#以随机选取的70%作为训练集，剩下的30%作为测试集,不对数据进行处理
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_imputer, Y, test_size=0.3, random_state=0) 


# In[126]:

#首先利用朴素贝叶斯进行分类
from sklearn.naive_bayes  import GaussianNB
model1 = GaussianNB()
model1.fit(train_x,train_y)
pred = model1.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# In[73]:

#利用随机森林进行分类
from sklearn.ensemble  import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=50)
model2.fit(train_x,train_y)
pred = model2.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# In[198]:

#利用K近邻进行分类
from sklearn.neighbors  import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors=12)
model3.fit(train_x,train_y)
pred = model3.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# In[548]:

#训练集、测试集的划分，利用77个病人的43个蛋白质作为分类标准，
from sklearn import cross_validation
Y=processed['PAM50 mRNA']
#以随机选取的70%作为训练集，剩下的30%作为测试集,利用处理过的数据（异常值删除，正则化，标准化）
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_norm, Y, test_size=0.3, random_state=0) 


# In[301]:

#利用处理过的数据再次进行朴素贝叶斯，发现效果有所提高
from sklearn.naive_bayes  import GaussianNB
model1 = GaussianNB()
model1.fit(train_x,train_y)
pred = model1.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# In[109]:

#利用处理过的数据再次进行随机森林，效果也有所提高
from sklearn.ensemble  import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=50)
model2.fit(train_x,train_y)
pred = model2.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# In[553]:

#利用处理过数据进行K近邻，发现效果没有提高
from sklearn.neighbors  import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors=12)
model3.fit(train_x,train_y)
pred = model3.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# In[75]:

## 各个蛋白质基因的重要性,利用随机森林的结果
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

importances = model2.feature_importances_
std = np.std([tree.feature_importances_ for tree in model2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# 对各个基因重要性进行排序
print("Feature ranking:")

for f in range(train_x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# 绘制图形
plt.figure()
plt.title("Feature importances")
plt.bar(range(train_x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train_x.shape[1]), indices)
plt.xlim([-1, train_x.shape[1]])
plt.show()


# # 第四问:如何判断Clinicaly（第二个文件）中已知的其他分类结果是否也用了PAM50 mRNA所用到的43个蛋白质基因？
# 

# In[140]:

## 数据载入，处理表头
data = pd.read_csv(dataset_path,header=0,index_col=0)
clinical = pd.read_csv(clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample
pam50 = pd.read_csv(pam50_proteins,header=0)
 
## 去除symbol、name两列无，3列重复，3列健康人数据
data.drop(['gene_symbol','gene_name','AO-A12D.01TCGA','C8-A131.01TCGA','AO-A12B.01TCGA','263d3f-I.CPTAC','blcdb9-I.CPTAC','c4155b-C.CPTAC'],axis=1,inplace=True)
 
## Change the protein data sample names to a format matching the clinical data set
data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)
 
## 数据转置
data = data.transpose()
## 对第二个文件进行表头处理
clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data.index],:]

## 将clinical中数据与77_cancer_proteomes中数据进行整合，放在一起
merged = data.merge(clinical,left_index=True,right_index=True)
 
## 更改名称方便识别
processed = merged
 
## 将第二问中用不到的clinical中部分数据进行删除
processed_numerical = processed.loc[:,[x for x in processed.columns if bool(re.search("NP_|XP_",x)) == True]]

#将第三个文件中出现的PAM50蛋白质与第一个文件对应进行筛选，得到有用的43个基因
processed_numerical_p50 = processed_numerical.ix[:,processed_numerical.columns.isin(pam50['RefSeqProteinID'])]
processed1=pd.DataFrame(processed_numerical_p50)
processed1.head(10)


# In[143]:

#缺失值的填补（用中位数填充）
imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
imputer = imputer.fit(processed_numerical_p50)
data_imputer = imputer.transform(processed_numerical_p50)

#数据Z-scale 标准化
from sklearn import preprocessing
data_scaled = preprocessing.scale(data_imputer)

#数据正则化处理
normalizer=preprocessing.normalize(data_scaled,norm='l2')
data_norm=pd.DataFrame(normalizer)
data_norm.head(10)


# # 用PAM50 mRNA所用的43个蛋白质作为分类标准，来对CN Clusters进行分类

# In[272]:

from sklearn import cross_validation
Y=processed['CN Clusters']
#以随机选取的70%作为训练集，剩下的30%作为测试集,利用处理过的数据（异常值删除，正则化，标准化）
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_norm, Y, test_size=0.3, random_state=0) 


# In[288]:

from sklearn.ensemble  import RandomForestClassifier
m5 = RandomForestClassifier(n_estimators=50)
m5.fit(train_x,train_y)
pred = m5.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


#  # 用PAM50 mRNA所用的43个蛋白质作为分类标准，来对SigClust Unsupervised mRNA进行分类

# In[307]:

from sklearn import cross_validation
Y=processed['SigClust Unsupervised mRNA']
#以随机选取的70%作为训练集，剩下的30%作为测试集,利用处理过的数据（异常值删除，正则化，标准化）
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_norm, Y, test_size=0.3, random_state=0) 


# In[318]:

from sklearn.ensemble  import RandomForestClassifier
m6 = RandomForestClassifier(n_estimators=50)
m6.fit(train_x,train_y)
pred = m6.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


#  # 用PAM50 mRNA所用的43个蛋白质作为分类标准，来对miRNA Clusters进行分类

# In[319]:

from sklearn import cross_validation
Y=processed['miRNA Clusters']
#以随机选取的70%作为训练集，剩下的30%作为测试集,利用处理过的数据（异常值删除，正则化，标准化）
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_norm, Y, test_size=0.3, random_state=0) 


# In[338]:

from sklearn.ensemble  import RandomForestClassifier
m7 = RandomForestClassifier(n_estimators=50)
m7.fit(train_x,train_y)
pred = m7.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# # 用PAM50 mRNA所用的43个蛋白质作为分类标准，来对methylation Clusters进行分类

# In[339]:

from sklearn import cross_validation
Y=processed['methylation Clusters']
#以随机选取的70%作为训练集，剩下的30%作为测试集,利用处理过的数据（异常值删除，正则化，标准化）
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_norm, Y, test_size=0.3, random_state=0) 


# In[357]:

from sklearn.ensemble  import RandomForestClassifier
m8 = RandomForestClassifier(n_estimators=50)
m8.fit(train_x,train_y)
pred = m8.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# # 用PAM50 mRNA所用的43个蛋白质作为分类标准，来对RPPA Clusters进行分类

# In[359]:

from sklearn import cross_validation
Y=processed['RPPA Clusters']
#以随机选取的70%作为训练集，剩下的30%作为测试集,利用处理过的数据（异常值删除，正则化，标准化）
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_norm, Y, test_size=0.3, random_state=0) 


# In[373]:

from sklearn.ensemble  import RandomForestClassifier
m9 = RandomForestClassifier(n_estimators=50)
m9.fit(train_x,train_y)
pred = m9.predict(test_x)
from sklearn import metrics
print(metrics.classification_report(test_y,pred))

print(metrics.accuracy_score(test_y,pred))


# In[ ]:



