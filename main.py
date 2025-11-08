import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report

data = pd.read_csv("student-performance.csv",sep=";")

#Data preparation
data=data.map(lambda x: x.strip() if isinstance(x, str) else x)
data=data.drop(columns=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","romantic","paid","famrel","Dalc","Walc"])
for col in data.columns:
    if(col in ["schoolsup","famsup","activities","nursery","higher","internet"]):
        data[col]=pd.Series((0 if val=="no" else 1) for val in data[col])
#Lets create a new atRisk column based off G3. Let 12/60 (60%) be the percentage required to pass.
data["atRisk"]=pd.Series((1 if grade<12 else 0) for grade in data["G3"])
#Lets drop the now un-necessary G3 column
data=data.drop("G3",axis=1)

#From Scratch implementation
def Sigmoid(X,w,b):
    return 1 / (1 + np.exp(-(np.dot(X,w) + b)))
def Cost(h, X, y):
    m = y.shape[0]
    ep = 1e-5 #to avoid taking log of zero
    #converting to numpy arrays
    h = np.array(h)
    y = np.array(y)
    X = np.array(X)
    sum = np.sum(y*np.log(h+ep) + (1-y)*np.log(1-h+ep))
    loss = -sum/m
   #  print(loss)
    dh_dw = (1/m)*(np.dot(X.T,(h-y)))
    dh_db = (1/m)*(np.sum(h-y))
    return dh_dw, dh_db
def Update(w,b,dh_dw,dh_db,a):
    w_n = w-a*dh_dw
    b_n = b-a*dh_db
    return w_n, b_n
def GradientDescent(X,y,a,n):
    w = np.zeros(X.shape[1])
    b=0
    dh_dw=0
    dh_db=0
    for i in range(n):
        h=Sigmoid(X,w,b)
        dh_dw,dh_db=Cost(h,X,y)
        w,b=Update(w,b,dh_dw,dh_db,a)
    return w,b
def LogisticRegression(X,w,b,c):
    result = Sigmoid(X,w,b)
    for i in range(len(result)):
      if result[i]>=c:
         result[i]=1
      else:
         result[i]=0
    return result

#Partition of datasets
X = data.drop(columns=["atRisk"])
Y=data["atRisk"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

w,b = GradientDescent(X_train, y_train, 0.05, 1000)
y_predFS = LogisticRegression(X_test, w, b, 0.575)

model = linear_model.LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
y_predSK = model.predict(X_test)

print("Comparison of predictions with actual values:")
output = pd.DataFrame({'Actual': y_test.values,
                       'Predicted_FS': y_predFS,
                       'Predicted_SK': y_predSK})
print(output.to_string())

resultFS = classification_report(y_test, y_predFS)
print("From scratch implementation:")
print(resultFS)
print("Scikit implementation:")
resultSK = classification_report(y_test, y_predSK)
print(resultSK)