import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df =pd.read_excel('106年新竹站_20180309.xls')#讀檔
df = df.replace('NR',0) #沒資料NR先填0
a = df.iloc[4914: , 3:] #10,11,12月

for i in range(4914,6570):
    for j in range(24):
        if type(a.ix[i, j]) is str :
            if j == 0:    
                k = 0
                while j+k < 24:
                    if type(a.ix[i, j+k]) != str  :
                        a.ix[i, j] = a.ix[i, j+k]
                        break 
                    k += 1
            else:
                k = 0
                while j+k < 24:
                    if type(a.ix[i, j+k]) != str  :
                        a.ix[i,j] = (a.ix[i,j+k]+a.ix[i,j-1])/2 #前方值加後面值/2
                        break
                    if type(a.ix[i, j-1]) != str  :
                        a.ix[i, j] = a.ix[i, j-1]
                    k += 1
a = a.fillna(method ='ffill')
a.index = range(len(a))

z = df.iloc[4914:,0:3]
z.index = range(len(z))
res = pd.concat([z,a],axis=1 )

M10 = res.iloc[:1098, 0:] #10,11月
M12 = res.iloc[1098:, 0:] #12月
M12.index = range(len(M12))

train = pd.pivot_table(M10,index=["測項"],columns=["日期"]) #train:10,11月
train = train.reorder_levels([1, 0], axis=1).sort_index(axis=1)
test = pd.pivot_table(M12,index=["測項"],columns=["日期"])  #test:12月
test = test.reorder_levels([1, 0], axis=1).sort_index(axis=1)


x18= [[]]*(61*24-6) #18種屬性:X
for i in range(61*24-6):
    x18[i]= np.array(train.iloc[:,i:i+6])
xx18 = []
for i in range(len(x18)):
    temp = x18[i]
    temp = np.reshape(temp,18*6)
    xx18.append(temp)


y18= [[]]*(61*24-6) #18種屬性:Y 
for i in range(61*24-6):
    y18[i]= np.array(train.iloc[:,i+6])


pmx= [[]]*(61*24-6) #只有PM2.5:X 
for i in range(61*24-6):
    pmx[i]= train.iloc[9,i:i+6]
pmy= [[]]*(61*24-6) #只有PM2.5:Y 
for i in range(61*24-6):
    pmy[i]= train.iloc[9,i+6]
    

total_testx = [[]]*(31*24-6) #12月total 0~5...
for i in range(31*24-6):
    total_testx[i]= np.array(test.iloc[:,i:i+6])
total_testxx = []
for i in range(len(total_testx)):
    temp = total_testx[i]
    temp = np.reshape(temp,18*6)
    total_testxx.append(temp)

total_testy = [[]]*(31*24-6) #12月total 6.
for i in range(31*24-6):
    total_testy[i]= test.iloc[:,i+6]


pm_testx = [[]]*(31*24-6) #12月PM2.5 0~5...
for i in range(31*24-6):
    pm_testx[i]= test.iloc[9,i:i+6]
pm_testy = [[]]*(31*24-6) #12月PM2.5 6....
for i in range(31*24-6):
    pm_testy[i]= test.iloc[9,i+6]

lm = LinearRegression()
lm.fit(pmx,pmy) #PM2.5 LinearRegression訓練模型
y_pred = lm.predict(pm_testx)
print("PM2.5之Linear Regression MAE: ",mean_absolute_error(pm_testy, y_pred))

regr = RandomForestRegressor()
regr.fit(pmx,pmy) #PM2.5 RandomForestRegressor訓練模型
y_pred = regr.predict(pm_testx)
print("PM2.5之Random Forest Regression MAE: ",mean_absolute_error(pm_testy, y_pred))

lm = LinearRegression()
lm.fit(xx18,pmy)  #total之Linear Regression
y_pred = lm.predict(total_testxx)
print("total之Linear Regression MAE: ",mean_absolute_error(pm_testy, y_pred))

regr = RandomForestRegressor()
regr.fit(xx18,y18)  #total之Random Forest Regression
y_pred = regr.predict(total_testxx)
print("total之Random Forest Regression MAE: ",mean_absolute_error(pm_testy, y_pred))