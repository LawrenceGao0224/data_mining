import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df =pd.read_csv('data.csv')#讀檔
#df = df.replace(' ?','misunderstand')
dummy = pd.get_dummies(df) #用dummy 的方法

a = df.iloc[:,-1]  
b = []
for i in range(len(a)): #將[income]轉成0,1 已便訓練
    if(a[i]==" <=50K"):
        b.append(1)
    else:
        b.append(0)
b = np.array(b)
c = 0
group = np.mod(np.arange(32561),100)//10  #32561筆資料，每100筆一個區分，test取10筆
for testg in range(10):              
    trainidx = np.where(group!=testg)[0] #train 的index
    testidx = np.where(group==testg)[0]  #test 的 index
    
    X_train = dummy.iloc[trainidx, :-2] 
    X_test =  dummy.iloc[testidx, :-2]
    Y_train = b[trainidx]
    Y_test =  b[testidx]
    clf = GradientBoostingClassifier() #以Gradient Boosting進行分類
    clf = clf.fit(X_train, Y_train) #進行訓練
    y_pred = clf.predict(X_test)
    c += accuracy_score(Y_test,y_pred) #正確率
print('Accuracy:  ',c/10)