import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neighbors import KNeighborsClassifier as K
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

d = pd.read_csv('knn_project')
scaler = SS()
scaler.fit(d.drop('TARGET CLASS',axis=1))
st= scaler.transform(d.drop('TARGET CLASS',axis=1))
df=pd.DataFrame(data=st,columns=d.columns[:-1])
X_train, x_test, y_train, y_test = train_test_split(df,d['TARGET CLASS'],test_size=0.30,random_state=80)



r=[]
j=0
small_k=0
mn=10e+7
while True:
    j+=1
    if j-small_k>30:
        break
    kn=K(n_neighbors=j)
    kn.fit(X_train,y_train)
    p=kn.predict(x_test)
    z=np.mean(p!=y_test)
    r.append(z)
    if mn!=min(mn,z):
        mn=z
        small_k=j

k=K(n_neighbors=small_k)
k.fit(X_train,y_train)
pred=k.predict(x_test)
print("Confusion Matrix for KNN model of K=",small_k)
print(confusion_matrix(y_test,pred))
print("Classification Matrix")
print(classification_report(y_test,pred))
plt.figure(figsize=(12,6))
sns.set_style('darkgrid')
plt.plot(r,ls='--',marker='o',markerfacecolor='red',markersize=5)
plt.xlabel('k values')
plt.ylabel('error rate')
plt.title('error vs k')
#sns.jointplot(x=y_test,y=pred,marker='o')
# plt.xlabel('ytest')
# plt.ylabel('predicted values')
# plt.title('KNN Classification Model with K='+str(small_k))
plt.show(block=False)
plt.pause(5)
plt.close()