import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df=pd.read_csv("alzheimers_disease_data.csv")
#--------------- EDA ----------------
print(df.info())
print(df.describe())
print(df.head())
df=df.drop("PatientID",axis=1)
df=df.drop("DoctorInCharge",axis=1)
print("*"*80)
#checking class balance
print("\nDiagnosis Distribution:")
print(df['Diagnosis'].value_counts())
#features and target
X=df.drop("Diagnosis",axis=1)
y=df['Diagnosis']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#------------------- Logistic Regression --------------------
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("*"*80)
print("LOGISTIC REGRESSION\n")
print("Accuracy: ", accuracy_score(y_test,y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report")
print(classification_report(y_test,y_pred))
print("*"*80)
#-------------------- K-Nearest Neighbors ------------------------
knn=KNeighborsClassifier(n_neighbors=5,
                         metric='euclidean')
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("K-NEAREST NEIGHBORS")
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report")
print(classification_report(y_test,y_pred))
print("*"*80)
#--------------------- Naive Bayes -----------------------------------
model2=GaussianNB()
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
print("NAIVE BAYES")
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report")
print(classification_report(y_test,y_pred))
print("*"*80)
#--------------------- Random Forest -----------------------------
rf=RandomForestClassifier(n_estimators=200,
                          random_state=42)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print("RANDOM FOREST")
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report")
print(classification_report(y_test,y_pred))
print("*"*80)













