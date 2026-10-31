import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df=pd.read_csv("alzheimers_disease_data.csv")
#--------------------- EDA ----------------------------------
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())
print("Duplicte Rows: ",df.duplicated().sum())
df=df.drop("PatientID",axis=1)
df=df.drop("DoctorInCharge",axis=1)
print("\nDiagnosis Distribution:")
Diagnosis_counts=df['Diagnosis'].value_counts()
print(Diagnosis_counts)
Diagnosis_counts.plot(
    kind='pie', 
    autopct='%1.1f%%', 
    #startangle=90, 
    figsize=(6, 6),
    colors=['skyblue', 'lightgreen', 'salmon']
)
plt.title("Class Distribution (0 = No Alzheimer's, 1 = Alzheimer's)")
plt.tight_layout()
plt.show()
plt.figure(figsize=(16, 12))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
print("Top correlations with Diagnosis")
print(corr["Diagnosis"].drop("Diagnosis").sort_values(ascending=False).head(10).round(3))
print("*"*80)
print("\nDiagnosis Distribution:")
print(df['Diagnosis'].value_counts())
cat_features = ["Gender", "Smoking", "FamilyHistoryAlzheimers", "Depression"]
for i, col in enumerate(cat_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, hue="Diagnosis", data=df, palette="Set2")
    plt.title(f"{col} vs Diagnosis")
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()
print("Lifestyle vs Diagnosis")
for col in ["BMI", "Smoking", "AlcoholConsumption",
            "PhysicalActivity", "DietQuality", "SleepQuality"]:
    print(f"{col}: {df.groupby('Diagnosis')[col].mean().round(2).to_dict()}")
print("Medical History vs Diagnosis")
med_cols = ["FamilyHistoryAlzheimers", "CardiovascularDisease",
            "Diabetes", "Depression", "HeadInjury", "Hypertension"]
for col in med_cols:
    rate = df.groupby(col)["Diagnosis"].mean().round(3)
    print(f"{col}: P(Alzheimers=1)={rate.to_dict()}")
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Age", hue="Diagnosis", kde=True, palette="Set2", bins=20)
plt.title("Age Distribution by Diagnosis")
plt.tight_layout()
plt.show()
leakage_cols = ["MMSE", "FunctionalAssessment", "ADL", "MemoryComplaints",
                "BehavioralProblems", "Confusion", "Disorientation",
                "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"]
df = df.drop(columns=leakage_cols)
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
#----------------------- SVM -------------------------------------
svm = SVC(
    kernel='rbf',
    random_state=42
)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("SUPPORT VECTOR MACHINE")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))










