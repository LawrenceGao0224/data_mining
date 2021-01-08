import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree, preprocessing
import graphviz
from sklearn.metrics import precision_score,recall_score, confusion_matrix,accuracy_score
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

df =pd.read_csv('character-deaths.csv')#讀檔
df = df.fillna(0) #沒資料填0
a = df.replace({'Book of Death':[1,2,3,4,5]},1) 
dummy = pd.get_dummies(df['Allegiances'])
df = pd.concat([a,dummy],axis=1)

X = df.iloc[:, 5:]
y = df.iloc[:, 3] #以 Book of death為y
seed =42 
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = seed)
#選曲訓練資料75,25

clf = tree.DecisionTreeClassifier(random_state=0) 
clf = clf.fit(X_train,y_train) #進行訓練
y_pred = clf.predict(X_test)

print('Precision: ',precision_score(y_test,y_pred))
print('Recall: ',recall_score(y_test,y_pred))
print('Accuracy: ',accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# Create DOT data
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,max_depth= 5)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Show graph
graph.write_pdf("Decision tree.pdf")