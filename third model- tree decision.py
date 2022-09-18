# decision tree
import pandas as pd
dataset= pd.read_csv("Social_Network_Ads.csv") #pandas to read the dataset
x= dataset.iloc[: , 2:4].values
y= dataset.iloc[:, 4].values
# scaling phase
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x= sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.30)
from sklearn.tree import DecisionTreeClassifier
dc= DecisionTreeClassifier()
dc.fit(x_train,y_train) #learining process
output = dc.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,output)
print(cm)
sum=0
for i in range(10): #avearge accuracy is constant but the accurcy varies
    ac = ((cm[0][0])+(cm[1][1]))/(len(y_test))
    sum+=ac
av= sum/10 ; print("acuureacy: ", av*100, "%")
